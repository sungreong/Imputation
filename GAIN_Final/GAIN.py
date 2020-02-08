import category_encoders as ce
import pandas as pd, dill
from copy import deepcopy
import numpy as np
from IPython.display import clear_output
import pandas as pd
import tensorflow as tf
from tqdm import tqdm, tqdm_notebook
import numpy as np, re, os
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings, dill, sys
import inspect , seaborn as sns
warnings.filterwarnings('ignore')
from Common import *

sys.path.append('/home/advice/Python/SR/Custom/')
from RAdam import RAdamOptimizer
from renom.utility import completion
import logging, datetime
import logging.handlers
import shutil


def ck_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    else:
        shutil.rmtree(dir_)
        os.makedirs(dir_)


def typecheck(data):
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise TypeError(f"invalid type : {type(data)}")
    return data


class GAIN(Com):
    """
    ### missing_info must save specific path
    missing_info (dict)
    >>> 1.obj_range : object range
    >>> 2.obj_info : object 1. start idx 2. n
    >>> 3.ce_one_hot : category encoder
    >>> 4.columns : total column
    >>> 5.scaler  : numerical scaler
    >>> 6,num_var : numeric column
    >>> 8.fac_var : object column
    """

    def __init__(self, gen_dim, dis_dim, information, gpu=1):
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim
        ########################################

        self.missing_info = information
        self.ce_one_hot = self.missing_info["ce"]
        self.columns = self.missing_info["train_onehot_col"]
        self.scaler = self.missing_info["scaler"]
        self.in_var = self.missing_info["in_var"]
        self.num_var = self.missing_info["num_var"]
        self.fac_var = self.missing_info["fac_var"]
        self.cond = self.missing_info["store"]
        self.key_cond = self.missing_info["key_store"]
        self.weight_info = self.missing_info["weight_info"]
        self.ForComp = self.missing_info["ForComp"]
        self.ForComp_missing_matrix = self.missing_info["ForComp_missing_matrix"]
        self.curr_iter = 0
        self.scale_range = self.missing_info["scaler"].feature_range
        if self.scale_range == (0, 1):
            self.G_final_Act = tf.nn.sigmoid
        elif self.scale_range == (-1, 1):
            self.G_final_Act = tf.nn.tanh

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    def update_dict_(self, dict):
        self.__dict__.update(dict)

    def generator(self, New_X):
        activation = self.Gact
        init = self.W_Init(activation)

        with tf.variable_scope("GAN/Generator"):
            inputs = tf.concat(axis=1, values=[New_X, self.M])  # Mask + Data Concatenate
            all_h = self.gen_dim
            if self.ck_max_norm:
                max_norm_reg = self.max_norm_regularizer(threshold=1.0)
            else:
                max_norm_reg = None
            for idx, h_dim in enumerate(self.gen_dim):
                if idx == 0:
                    GAN_Weight = tf.get_variable(f"GWeight_{idx}", dtype=tf.float32,
                                                 shape=[self.Dim * 2, h_dim],
                                                 regularizer=max_norm_reg,
                                                 initializer=init)
                    if self.ck_SN:
                        GAN_Weight = self.spectral_norm(GAN_Weight,
                                                        name="G_SN_Weight" + str(idx))
                    else:
                        pass
                    GAN_bias = tf.get_variable(f"GBias_{idx}",
                                               shape=[h_dim], dtype=tf.float32,
                                               initializer=tf.constant_initializer(0.0))
                    Layer = self.Gact(tf.matmul(inputs, GAN_Weight) + GAN_bias)
                    if self.Gact == tf.nn.selu:
                        Layer = tf.contrib.nn.alpha_dropout(Layer, self.dropoutRate)
                else:
                    GAN_Weight = tf.get_variable(f"GWeight_{idx}", dtype=tf.float32,
                                                 shape=[all_h[idx - 1], all_h[idx]],
                                                 regularizer=max_norm_reg,
                                                 initializer=init)
                    if self.ck_SN:
                        GAN_Weight = self.spectral_norm(GAN_Weight,
                                                        name=f"G_SN_Weight{idx}")
                    else:
                        pass
                    GAN_bias = tf.get_variable(f"GBias_{idx}",
                                               shape=[all_h[idx]], dtype=tf.float32,
                                               initializer=tf.constant_initializer(0.0))
                    Layer = self.Gact(tf.matmul(Layer, GAN_Weight) + GAN_bias)
                    if self.Gact == tf.nn.selu:
                        Layer = tf.contrib.nn.alpha_dropout(Layer, self.dropoutRate)

            activation = self.G_final_Act
            init = self.W_Init(activation)
            GAN_Weight = tf.get_variable("GWeight_Final", dtype=tf.float32,
                                         shape=[self.gen_dim[-1], self.Dim],
                                         initializer=init)
            if self.ck_SN:
                GAN_Weight = self.spectral_norm(GAN_Weight, name="G_SN_Weight_Final")
            else:
                pass
            GAN_bias = tf.get_variable("GBias_Final", shape=[self.Dim], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
            Logit = tf.matmul(Layer, GAN_Weight) + GAN_bias
            output, onehot, argmax = self.CatNumEmb(Logit, self.cond, self.key_cond)
            return [Logit, output, onehot, argmax]

    def discriminator(self, X, Hint):
        activation = self.Dact
        init = self.W_Init(activation)

        if self.ck_max_norm:
            max_norm_reg = self.max_norm_regularizer(threshold=1.0)
        else:
            max_norm_reg = None
        with tf.variable_scope("GAN/Discriminator"):
            inputs = tf.concat(axis=1, values=[X, Hint])  # Hint + Data Concatenate
            all_h = self.dis_dim
            for idx, h_dim in enumerate(self.dis_dim):
                if idx == 0:
                    Dis_Weight = tf.get_variable(f"DWeight_{idx}",
                                                 dtype=tf.float32,
                                                 shape=[self.Dim * 2, h_dim],
                                                 regularizer=max_norm_reg,
                                                 initializer=init)
                    if self.ck_SN:
                        Dis_Weight = self.spectral_norm(Dis_Weight,
                                                        name=f"D_SN_Weight{idx}")
                    else:
                        pass
                    Dis_bias = tf.get_variable(f"DBias_{idx}",
                                               shape=[h_dim], dtype=tf.float32,
                                               initializer=tf.constant_initializer(0.0001))
                    Layer = self.Dact(tf.matmul(inputs, Dis_Weight) + Dis_bias)
                    if self.Dact == tf.nn.selu:
                        Layer = tf.contrib.nn.alpha_dropout(Layer, self.dropoutRate)
                else:
                    Dis_Weight = tf.get_variable(f"DWeight_{idx}",
                                                 dtype=tf.float32,
                                                 shape=[all_h[idx - 1], all_h[idx]],
                                                 regularizer=max_norm_reg,
                                                 initializer=init)
                    if self.ck_SN:
                        Dis_Weight = self.spectral_norm(Dis_Weight,
                                                        name=f"D_SN_Weight{idx}")
                    else:
                        pass
                    Dis_bias = tf.get_variable(f"DBias_{idx}",
                                               shape=[all_h[idx]],
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(0.0001))
                    Layer = tf.matmul(Layer, Dis_Weight) + Dis_bias
                    Layer = self.Dact(Layer)
                    if self.Dact == tf.nn.selu:
                        Layer = tf.contrib.nn.alpha_dropout(Layer, self.dropoutRate)

            init = self.W_Init(activation)

            Dis_Weight = tf.get_variable("DWeight_Final", dtype=tf.float32,
                                         shape=[self.dis_dim[-1], self.Dim],
                                         initializer=init)
            if self.ck_SN:
                Dis_Weight = self.spectral_norm(Dis_Weight,
                                                name="D_SN_Weight_Final")
            else:
                pass
            Dis_bias = tf.get_variable("DBias_Final",
                                       shape=[self.Dim], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
            Logit = tf.matmul(Layer, Dis_Weight) + Dis_bias
            Output = tf.nn.sigmoid(Logit)
            return Output

    def define(self, ):
        # 1.1. Data Vector
        self.X = tf.placeholder(tf.float32, shape=[None, self.Dim], name="true_data")
        # 1.2. Mask Vector  (미싱 아닌 부분 표시된 것)
        self.M = tf.placeholder(tf.float32, shape=[None, self.Dim], name="mask_vector")
        # 1.3. Hint vector
        self.H = tf.placeholder(tf.float32, shape=[None, self.Dim])
        # 1.4. X with missing values
        self.New_X = tf.placeholder(tf.float32, shape=[None, self.Dim], name="missing_data")
        self.dropoutRate = tf.placeholder(tf.float32, name="dropoutRate")
        self.steps = []
        self.pfc_store = []
        self.rmse_store = []
        self.gen_loss = []
        self.disc_loss = []
        self.train_loss_tr = []
        self.train_loss_te = []
        self.test_loss_tr = []
        self.test_loss_te = []
        self.sep_store = []

    def set_env(self, TrainSet, ValidSet, mb_size, hint,
                Gact=tf.nn.selu, Dact=tf.nn.selu, alpha=3,
                patience=10, cutoff=0.00001, lr=0.001,
                epoch=1000, weight_regularizer=0.005, max_norm=True, SN=False):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        del values["frame"]
        self.env = values

    def fit(self, ):
        self.update_dict_(self.env)
        self.update_dict_(self.__dict__)
        select_w_init = np.random.randint(0, 2, size=1)[0]
        seed_n = np.random.randint(1, 1000, size=1)[0]
        #         self.patience = patience
        #         self.cut_off = cutoff
        #         self.ck_max_norm = max_norm
        #         self.ck_SN = SN
        #         self.Gact = Gact
        #         self.Dact = Dact
        #         self.epoch = epoch + 1
        #         trainX, trainM = TrainSet
        #         testX, testM = ValidSet
        #         self.p_hint = hint
        #         self.mb_size = mb_size
        #         self.alpha = alpha
        self.relu_w_init = [tf.keras.initializers.he_uniform(seed=seed_n),
                            tf.keras.initializers.he_normal(seed=seed_n)][select_w_init]
        self.tanh_w_init = [tf.keras.initializers.glorot_normal(seed=seed_n),
                            tf.keras.initializers.glorot_uniform(seed=seed_n)][select_w_init]
        self.s_elu_w_init = [tf.keras.initializers.lecun_normal(seed=seed_n),
                             tf.keras.initializers.lecun_uniform(seed=seed_n)][select_w_init]
        self.nomal_w_init = tf.keras.initializers.truncated_normal(seed=seed_n)
        self.ck_max_norm = self.max_norm
        self.ck_SN = self.SN
        self.p_hint = self.hint
        weight_regularizer = self.weight_regularizer
        lr = self.lr
        trainX, trainM = self.TrainSet
        testX, testM = self.ValidSet
        self.trainX = typecheck(trainX)
        ##################### Matrix는 1-미싱 => 미싱 아닌 부분
        self.trainM = typecheck(1 - 1 * trainM)
        self.testX = typecheck(testX)
        self.testM = typecheck(1 - 1 * testM)
        self.total_X = np.concatenate((trainX, testX))
        self.total_M = np.concatenate((self.trainM, self.testM))
        self.Train_No, self.Dim = self.trainX.shape
        self.total = self.Dim

        ## modeling
        tf.reset_default_graph()
        self.define()
        ## M 은 미싱이 아닌 것!
        ## 미싱이 부분에 진짜를 가짜인 부분에 생성된 것을
        result = self.generator(self.New_X)
        Logit, G_sample, OnehotResult, ArgResult = result

        if self.fac_var == []:
            for v, col in enumerate(self.in_var):
                value = tf.slice(G_sample, [0, v], [-1, 1])  #
                tf.summary.histogram("Input_" + col.replace(" ", "_"), value)
        else:
            self.ArgResult = tf.identity(ArgResult, name="Arg_G")
            for v, col in enumerate(self.in_var):
                value = tf.slice(ArgResult, [0, v], [-1, 1])  #
                tf.summary.histogram("Input_" + col.replace(" ", "_"), value)
        Hat_New_X = self.M * self.New_X + (1 - self.M) * G_sample

        ## M은 미싱인 부분
        # imputed = self.New_X * (1-self.M) + G_sample * self.M
        self.Hat_New_X = tf.identity(Hat_New_X, name="imputed")
        self.G_sample = tf.identity(G_sample, name="generated")
        # Discriminator
        D_prob = self.discriminator(Hat_New_X, self.H)
        t_vars = tf.trainable_variables()
        if weight_regularizer > 0:
            G_L2 = []
            D_L2 = []
            for v in t_vars:
                if re.search('Weight', v.name):
                    if re.search("Generator", v.name):
                        print("G : ", v.name)
                        G_L2.append(tf.nn.l2_loss(v))
                    elif re.search("Discriminator", v.name):
                        print("D : ", v.name)
                        D_L2.append(tf.nn.l2_loss(v))
            self.Generator_W_l2 = tf.add_n(G_L2) * weight_regularizer
            self.Discriminator_W_l2 = tf.add_n(D_L2) * weight_regularizer
        else:
            self.Generator_W_l2 = tf.constant(0.0)
            self.Discriminator_W_l2 = tf.constant(0.0)
        for var in t_vars:
            tf.summary.histogram(var.op.name, var)
        self.D_1 = - self.M * tf.log(D_prob + 1e-8)
        self.D_2 = - (1 - self.M) * tf.log(1. - D_prob + 1e-8)
        self.D_3 = tf.reduce_mean(self.D_1 + self.D_2)
        self.D_loss = self.D_3 + self.Discriminator_W_l2
        self.G_loss1 = -tf.reduce_mean((1 - self.M) * tf.log(D_prob + 1e-8))
        ## 미싱이 아닌 부분 -> 미싱 부분
        if self.fac_var == []:
            Logit = self.G_final_Act(Logit)
        else:
            pass
        self.MSE_train_loss = self.CatNumEmb_Loss(Logit, self.X, self.M,
                                                  self.cond, self.key_cond, self.weight_info)
        #         self.MSE_train_loss_2 =self.CatNumEmb_Loss(Logit , self.X , self.M , seg = "Test")
        self.G_loss = \
            self.G_loss1 + self.alpha * self.MSE_train_loss + self.Generator_W_l2
        #         self.MSE_test_loss =\
        #         tf.reduce_mean( tf.square( (1-self.M) * self.X - (1-self.M) * G_sample ) )

        with tf.variable_scope("Original/Loss"):
            tf.summary.scalar("Total_G_loss", self.G_loss)
            tf.summary.scalar("Not_Missing_Loss", self.MSE_train_loss)
            tf.summary.scalar("D_Loss", self.D_loss)

        self.clip_all_weights = tf.get_collection("max_norm")
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
        disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")
        self.global_step = tf.get_variable('global_step', [],
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.lr = tf.train.cosine_decay_restarts(lr, self.global_step,
                                                 first_decay_steps=100,
                                                 t_mul=1.2, m_mul=0.95, alpha=0.5)
        self.D_solver = RAdamOptimizer(learning_rate=self.lr,
                                       beta1=0.5, beta2=0.5,
                                       weight_decay=0.0).minimize(self.D_loss, var_list=disc_vars)
        self.G_solver = tf.train.RMSPropOptimizer(learning_rate=self.lr, ).minimize(self.G_loss, var_list=gen_vars)
        comment = "{} \n{}{}{}\n{}".format("=" * 56, " " * 24, "모델피팅", " " * 24, "=" * 56)
        return print(comment)

    def train(self, save_model_path):
        import tensorflow
        ck_dir(save_model_path)
        self.save_model_path = save_model_path
        ##########################
        self.log = logging.getLogger("")
        if (self.log.hasHandlers()): self.log.handlers.clear()
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = True
        formatter = logging.Formatter("| %(asctime)s | %(message)s",
                                      "%Y-%m-%d %H:%M:%S")
        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(logging.DEBUG)
        streamHandler.setFormatter(formatter)
        self.log.addHandler(streamHandler)
        ##########################
        if self.curr_iter == 0:
            it = 0
            ck = "no"
            te_minumum = -1
            tr_minumum = -1
            self.lr_store = []
            config = tf.ConfigProto(log_device_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
        else:
            print("현재 Epoch : {}까지 진행하였습니다.".format(self.curr_iter))
            num = int(input("추가로 학습을 얼마나 진행하시겠습니까?\n >> "))
            self.epoch += num
            it = self.curr_iter
            ck, tr_minumum, te_minumum = self.ck, self.tr_minumum, self.te_minumum

        writer = tf.summary.FileWriter(self.save_model_path)
        writer.add_graph(self.sess.graph)
        merged_summary = tf.summary.merge_all()
        saver = tf.train.Saver()
        testN, test_dim = self.testX.shape
        test_Z_mb = self.sample_Z(testN, test_dim)
        test_New_X_mb = self.testM * self.testX + (1 - self.testM) * test_Z_mb
        H_mb1 = self.sample_M(testN, self.Dim, 1 - self.p_hint)
        test_H_mb = self.testM * H_mb1 + 0.5 * (1 - H_mb1)
        count = 0
        for it in range(it, self.epoch):
            self.printProgress(it, self.epoch)
            print("Epoch : {} 진행중...".format(it))
            idx = np.random.permutation(self.Train_No)
            XX = self.trainX[idx, :]
            MM = self.trainM[idx, :]
            batch_iter = int(self.Train_No / self.mb_size)
            Z_mb = self.sample_Z(self.mb_size, self.Dim)
            Gloss = []
            Dloss = []
            tr_non_missing_loss = []
            tr_missing_loss = []
            self.lr_store.append(self.sess.run(self.lr, feed_dict={self.global_step: it}))
            for idx in range(batch_iter):
                X_mb = XX[idx * self.mb_size:(idx + 1) * self.mb_size]
                M_mb = MM[idx * self.mb_size:(idx + 1) * self.mb_size]
                H_mb1 = self.sample_M(self.mb_size, self.Dim, 1 - self.p_hint)
                ## 미싱에 힌트를 넣어주는 부분
                H_mb = M_mb * H_mb1 + 0.5 * (1 - H_mb1)
                ## 진짜 인 것을 미싱에 넣고 가짜 인 것을 임의로 만들기
                New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce
                FeedDict = {self.M: M_mb, self.New_X: New_X_mb, self.X: X_mb,
                            self.H: H_mb,
                            self.global_step: it,
                            self.dropoutRate: 0.05
                            }
                d_step, g_step = 5, 1
                for _ in range(d_step):
                    _, D_loss_curr, D_3, Weight_D = \
                        self.sess.run([self.D_solver, self.D_loss,
                                       self.D_3, self.Discriminator_W_l2],
                                      feed_dict=FeedDict)

                if self.ck_max_norm: self.sess.run(self.clip_all_weights)

                for _ in range(g_step):
                    (_, G_loss_curr, MSE_train_loss_curr, G_1, Weight_G) = \
                        self.sess.run([
                            self.G_solver, self.G_loss, self.G_loss1,
                            self.MSE_train_loss, self.Generator_W_l2],
                            feed_dict=FeedDict)
                    if self.ck_max_norm:
                        self.sess.run(self.clip_all_weights)
                Dloss.append(D_loss_curr)
                Gloss.append(G_loss_curr)
                tr_non_missing_loss.append(MSE_train_loss_curr)
            #             try :
            #                 summary_str = self.sess.run(merged_summary ,  feed_dict = FeedDict)
            #                 writer.add_summary(summary_str , global_step = it)
            #             except Exception as e :
            #                 self.log.error(e)

            D_loss_curr, G_loss_curr = np.mean(Dloss), np.mean(Gloss)
            pfc_result, rmse_result = self.evaluate()
            self.steps.append(it)
            self.pfc_store.append(pfc_result)
            self.rmse_store.append(rmse_result)
            self.gen_loss.append(G_loss_curr)
            self.disc_loss.append(D_loss_curr)
            self.curr_iter = it
            self.ck, self.tr_minumum, self.te_minumum = ck, tr_minumum, te_minumum

            if (it % 5 == 0) & (it > 0):
                clear_output()
                plt.style.use('dark_background')
                fig, axes = plt.subplots(1, 2, figsize=(15, 3))
                plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                                    top=0.86, hspace=0, wspace=0.09)
                ax = axes.flatten()
                pfc = self.comparision["pfc"]
                rmse = self.comparision["rmse"]
                msg = \
                    f'PFC : {min(self.pfc_store):.3f} < {pfc:.3f} / RMSE : {min(self.rmse_store):.3f} < {rmse:.3f}'
                fig.get_axes()[0].annotate(msg, (0.5, 0.92),
                                           xycoords='figure fraction', ha='center',
                                           fontsize=15
                                           )
                ax[0].plot(self.steps, self.pfc_store)
                ax[0].hlines(pfc, 0, max(self.steps), colors="red")
                # ax[0].set_title(f"PFC : min : {min(self.pfc_store)}")
                ax[1].plot(self.steps, self.rmse_store)
                ax[1].hlines(rmse, 0, max(self.steps), colors="red")
                save_png = os.path.join(self.save_model_path, "rmse_pfc_result.png")
                plt.savefig(save_png)
                # ax[1].set_title(f"RMSE : min : {min(self.rmse_store)}")
                plt.close()
                save = [self.steps, self.pfc_store, self.rmse_store]
                save_file_path = os.path.join(self.save_model_path, "rmse_pfc_result.pkl")
                with open(save_file_path, "wb") as wb:
                    dill.dump(save, wb)

        #                 self.vis()
        #                 plt.title("[ Epoch : {} ] GLoss : {:.6f} DLoss : {:.6f}" \
        #                           .format(it, G_loss_curr, D_loss_curr),
        #                           fontsize=20)
        #                 png_file = os.path.join(self.save_model_path, "LossPlot.png")
        #                 plt.savefig(png_file)
        #                 plt.show()
        #                 if it % 10 == 0:
        #                     fig, ax = plt.subplots(figsize=(15, 3))
        #                     plt.subplots_adjust(left=0.18, bottom=0.1, right=0.95,
        #                                         top=0.95, hspace=0, wspace=0.2)
        #                     plt.plot(np.arange(len(self.lr_store)), self.lr_store)
        #                     plt.show()
        comment = "{} \n      {} \n{}".format("=" * 20, "학습완료", "=" * 20)
        return print(comment)

    def evaluate(self, ):
        trainZ = self.sample_Z(self.Train_No, self.Dim)
        trainH = self.sample_M(self.Train_No, self.Dim, 1 - self.p_hint)
        TotalFeed = {self.X: self.trainX,
                     self.M: self.trainM,
                     self.New_X: self.trainM * self.trainX + (1 - self.trainM) * trainZ,
                     self.H: trainH,
                     self.dropoutRate: 0.05}
        # want2see =[self.Hat_New_X,self.G_sample]
        result = self.sess.run(self.ArgResult, feed_dict=TotalFeed)
        result = pd.DataFrame(result, columns=self.in_var)
        result["SEX"] = result["SEX"] + 1
        imputed_result = (
                result * self.ForComp_missing_matrix +
                self.ForComp.fillna(-100) * (1 - self.ForComp_missing_matrix))
        self.visualize(self.key_cond , TotalFeed)
        pfc, rmse = self.calculate_rmse_pfc(self.not_missing_data,
                                            imputed_result,
                                            self.ForComp_missing_matrix == 1,
                                            self.in_var,
                                            self.num_var,
                                            self.fac_var)
        return pfc, rmse

    def running(self, path, init):
        tf.set_random_seed(init)
        self.fit()
        self.train(save_model_path=path)

    def run_parallel(self, title, n_count):
        from joblib import Parallel, delayed, cpu_count
        inits = [500, 1000, 1234, 2345, 3456]
        with Parallel(n_jobs=n_count, backend="multiprocessing") as parallel:
            results = parallel(
                delayed(self.running)(path=f"./{title}/{i}", init=inits[i]) for i in range(n_count))


    def visualize(self , key_cond , feed_dict):
        keys = list(key_cond.keys())
        mod = sys.modules[__name__]
        name = ['loss_{}'.format(c) for c in keys]
        # seper_loss = [getattr(mod, c) for c in name]
        losses = self.sess.run(self.seperate_var , feed_dict=feed_dict)
        self.sep_store.append(np.array(losses))
        vis = np.array(self.sep_store)
        default_dashes = \
            [() for _ in range(len(name))]
        index = np.arange(0, vis.shape[0])
        wide_df = pd.DataFrame(vis, index, name)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
        plt.style.use('dark_background')
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95,
                            top=0.95, wspace=None, hspace=0.0)
        palette = sns.color_palette("Spectral", len(name))
        sns.lineplot(data=wide_df,
                     dashes=default_dashes,
                     palette=palette)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        #                     ax.legend(loc="center right")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.005),
                  fancybox=True, shadow=True, ncol=5)
        save_png = os.path.join(self.save_model_path, "loss_mse_plot_{}.png".format(len(name)))
        plt.savefig(save_png)
        plt.close()
        #losses = " | ".join("{:.3f}".format(i) for i in losses)



from functools import wraps
import datetime
import time


def errorcheck(original_function):
    import logging
    import traceback
    import sys
    log = logging.getLogger('Error')
    log.setLevel(logging.DEBUG)
    log.propagate = True
    formatter = logging.Formatter("Error : %(asctime)s;\n%(message)s", "[%Y/%m/%d] %H:%M:%S")
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.addHandler(streamHandler)

    @wraps(original_function)  # 1
    def wrapper(*args, **kwargs):
        try:
            return original_function(*args, **kwargs)
        except Exception as e:

            exc_type, exc_obj, tb = sys.exc_info()
            formatted_lines = traceback.format_exc().splitlines()
            num = ([idx for idx, i
                    in enumerate(formatted_lines)
                    if re.search(" line ", i) is not None][-1])
            s = [line.split(",") for line in formatted_lines[num:]]
            import itertools
            merged = list(itertools.chain(*s))
            finalerror = "\n".join([string.strip() for string in merged])
            func_name = original_function.__name__
            errors = traceback.extract_stack()
            errors = ("".join(
                [
                    f"Where : {str(i).split('FrameSummary file ')[1].split(',')[0]} \n Line : {str(i).split('FrameSummary file ')[1].split(',')[1]}\n {'--' * 30} \n"
                    for i in errors[:-1]]))
            print(f"{'*' * 15} Search {'*' * 15} \n {errors}")
            log.error(f"{'==' * 20}\n{finalerror}\n{'==' * 20}")
            sys.exit(1)

    return wrapper
