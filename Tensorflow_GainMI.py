import numpy as np
from IPython.display import clear_output
import pandas as pd
import tensorflow as tf
from tqdm import tqdm , tqdm_notebook
import numpy as np
import re , os
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings , dill
from Common import Com
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('/home/advice/Python/SR/Custom/')
from RAdam import RAdamOptimizer
from renom.utility import completion
import logging , datetime
import logging.handlers
import shutil
from Common import focal_loss_sigmoid , focal_loss_softmax , tf_mish

def ck_dir(dir_) :
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    else :
        shutil.rmtree(dir_)
        os.makedirs(dir_)

"""
===============================================
  Common 에다가 사용하는 공통 Class를 모아놓음
===============================================
"""

class Missing_GAIN(Com) :
    """
    ### missing_info must save specific path
    missing_info (dict)
    >>> 1.obj_range : object range 
    >>> 2.obj_info : object 1. start idx 2. n
    >>> 3.ce_one_hot : category encoder
    >>> 4.columns : total column
    >>> 5.scaler  : numerical scaler
    >>> 6,num_col : numeric column
    >>> 7.ord_col : ordinal column
    >>> 8.obj_col : object column
    """
    def __init__(self, gen_dim , dis_dim , missing_info ,
                 save_info_path = None ,
                 save_model_path = './CAT_AE/Model' , gpu = 1) :
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim
        ck_dir(save_model_path)
        self.save_model_path  = save_model_path 
        ########################################
        try :
            print("{}에 missing info 저장".format(save_info_path))
            with open(save_info_path, 'wb') as f:
                dill.dump(missing_info , f)
        except Exception as e :
            print("Error : " ,  e)
            raise
        self.missing_info = missing_info
        self.obj_range = self.missing_info["obj_range"]
        self.obj_info = self.missing_info["obj_info"]
        self.ce_one_hot = self.missing_info["ce_encoder"]
        self.columns= self.missing_info["columns"]
        self.scaler = self.missing_info["scaler"]
        self.num_col = self.missing_info["num_col"]
        self.ord_col = self.missing_info["ord_col"]
        self.obj_col = self.missing_info["obj_col"]
        self.na_add_col = self.missing_info["ce_replace"]
        self.cond = self.missing_info["cat_num_idx_info"]
        self.Original_Column = self.missing_info["original_column"]
        self.curr_iter = 0
        self.scale_range = missing_info["scaler"].feature_range
        if self.scale_range == (0,1) :
            self.G_final_Act = tf.nn.sigmoid
        elif self.scale_range == (-1,1) :
            self.G_final_Act = tf.nn.tanh
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
        


    def generator(self , New_X) :
        activation = self.Gact
        init = self.W_Init(activation)
        
        with tf.variable_scope("GAN/Generator"):
            inputs = tf.concat(axis = 1, values = [New_X , self.M])  # Mask + Data Concatenate
            all_h = self.gen_dim
            if self.ck_max_norm : max_norm_reg = self.max_norm_regularizer(threshold = 1.0)
            else : max_norm_reg = None
            for idx , h_dim in enumerate(self.gen_dim) :
                if idx == 0 :
                    GAN_Weight = tf.get_variable("GWeight_" + str(idx) , dtype = tf.float32 , 
                                                 shape = [self.Dim * 2 , h_dim] , 
                                                 regularizer = max_norm_reg ,
                                                 initializer = init)
                    if self.ck_SN : GAN_Weight = self.spectral_norm(GAN_Weight, 
                                                                    name ="G_SN_Weight"+str(idx))
                    else : pass
                    GAN_bias = tf.get_variable("GBias_" + str(idx) , 
                                               shape = [h_dim] , dtype = tf.float32 , 
                                               initializer = tf.constant_initializer(0.0))
                    Layer = self.Gact(tf.matmul( inputs , GAN_Weight) + GAN_bias)
                    if self.Gact  == tf.nn.selu    :
                        Layer = tf.contrib.nn.alpha_dropout(Layer , self.dropoutRate ) 
                else :
                    GAN_Weight = tf.get_variable("GWeight_" + str(idx) , dtype = tf.float32 , 
                                                 shape = [all_h[idx-1] , all_h[idx]] , 
                                                 regularizer = max_norm_reg ,
                                                 initializer = init)
                    if self.ck_SN : GAN_Weight = self.spectral_norm(GAN_Weight, 
                                                                    name ="G_SN_Weight"+str(idx))
                    else : pass
                    GAN_bias = tf.get_variable("GBias_" + str(idx) , 
                                               shape = [all_h[idx]] , dtype = tf.float32 , 
                                               initializer = tf.constant_initializer(0.0))
                    Layer = self.Gact(tf.matmul( Layer , GAN_Weight) + GAN_bias)
                    if self.Gact  == tf.nn.selu    :
                        Layer = tf.contrib.nn.alpha_dropout(Layer , self.dropoutRate ) 
                        
            activation = self.G_final_Act
            init = self.W_Init(activation)
            GAN_Weight = tf.get_variable("GWeight_Final" , dtype = tf.float32 , 
                                         shape = [self.gen_dim[-1] , self.Dim] , 
                                         initializer = init)
            if self.ck_SN : 
                GAN_Weight = self.spectral_norm(GAN_Weight, 
                                                name ="G_SN_Weight_Final")
            else : pass
            GAN_bias = tf.get_variable("GBias_Final" , shape = [self.Dim] , dtype = tf.float32 , 
                                 initializer = tf.constant_initializer(0.0))
            Logit = tf.matmul( Layer , GAN_Weight) + GAN_bias
            if self.obj_col == [] :
                output = self.G_final_Act(Logit)
                argmax = None    
            else :
                output , argmax = self.CatNumEmb(Logit)
                
            return [Logit , output , argmax]

    def discriminator(self, X , Hint) :
        activation = self.Dact
        init = self.W_Init(activation)
        
        if self.ck_max_norm : max_norm_reg = self.max_norm_regularizer(threshold = 1.0 )
        else : max_norm_reg = None
        with tf.variable_scope("GAN/Discriminator"):
            inputs = tf.concat(axis = 1, values = [X,Hint])  # Hint + Data Concatenate
            all_h = self.dis_dim
            for idx , h_dim in enumerate(self.dis_dim) :
                if idx == 0 :
                    Dis_Weight = tf.get_variable("DWeight_" + str(idx) , 
                                                 dtype = tf.float32 , 
                                                 shape = [self.Dim * 2 , h_dim] , 
                                                 regularizer = max_norm_reg ,
                                                 initializer = init)
                    if self.ck_SN : Dis_Weight = self.spectral_norm(Dis_Weight, 
                                                                    name ="D_SN_Weight"+str(idx))
                    else : pass
                    Dis_bias = tf.get_variable("DBias_" + str(idx) , 
                                               shape = [h_dim] ,dtype = tf.float32 , 
                                               initializer = tf.constant_initializer(0.0001))
                    Layer = self.Dact(tf.matmul( inputs , Dis_Weight) + Dis_bias)
                    if self.Dact  == tf.nn.selu    :
                        Layer = tf.contrib.nn.alpha_dropout(Layer , self.dropoutRate ) 
                else :
                    Dis_Weight = tf.get_variable("DWeight_" + str(idx) , 
                                                 dtype = tf.float32 , 
                                                 shape = [all_h[idx-1] , all_h[idx]] , 
                                                 regularizer = max_norm_reg ,
                                                 initializer = init)
                    if self.ck_SN : Dis_Weight = self.spectral_norm(Dis_Weight, 
                                                                    name ="D_SN_Weight"+str(idx))
                    else : pass
                    Dis_bias = tf.get_variable("DBias_" + str(idx) , 
                                               shape = [all_h[idx]] , 
                                               dtype = tf.float32 , 
                                              initializer = tf.constant_initializer(0.0001))
                    Layer = tf.matmul( Layer , Dis_Weight) + Dis_bias
                    Layer = self.Dact(Layer)
                    if self.Dact  == tf.nn.selu    :
                        Layer = tf.contrib.nn.alpha_dropout(Layer , self.dropoutRate ) 
            
            
            Factivation = tf.nn.sigmoid
            init = self.W_Init(activation)
            
            Dis_Weight = tf.get_variable("DWeight_Final" , dtype = tf.float32 , 
                                         shape = [ self.dis_dim[-1] , self.Dim] , 
                                         initializer =init)
            if self.ck_SN : 
                Dis_Weight = self.spectral_norm(Dis_Weight, 
                                                name ="D_SN_Weight_Final")
            else : pass
            Dis_bias = tf.get_variable("DBias_Final" , 
                                       shape = [self.Dim] , dtype = tf.float32 , 
                                       initializer = tf.constant_initializer(0.0))
            Logit = tf.matmul( Layer , Dis_Weight) + Dis_bias
            Output = Factivation(Logit)
            return Output

    def sample_M(self, m, n, p):
        A = np.random.uniform(0., 1., size = [m, n])
        B = A > p
        C = 1.*B
        return C

    def fit(self , trainX , trainM ,testX ,testM   , mb_size ,  hint , 
            Gact = tf.nn.selu , Dact = tf.nn.selu , alpha = 3 ,
            patience = 10, cutoff = 0.00001 , lr = 0.001,
            epoch = 1000 , weight_regularizer = 0.005 ,max_norm = True , SN = False) :
        
        select_w_init = np.random.randint(0, 2, size=1 )[0]
        seed_n = np.random.randint(1, 1000, size=1 )[0]
        self.patience = patience
        self.cut_off = cutoff
        self.ck_max_norm = max_norm
        self.ck_SN = SN
        self.relu_w_init = [tf.keras.initializers.he_uniform(seed = seed_n) ,
                       tf.keras.initializers.he_normal(seed = seed_n)][select_w_init]
        self.tanh_w_init = [tf.keras.initializers.glorot_normal(seed = seed_n) ,
                       tf.keras.initializers.glorot_uniform(seed = seed_n)][select_w_init]
        self.s_elu_w_init = [tf.keras.initializers.lecun_normal(seed = seed_n) ,
                       tf.keras.initializers.lecun_uniform(seed = seed_n)][select_w_init]
        self.nomal_w_init = tf.keras.initializers.truncated_normal(seed = seed_n)
        self.Gact = Gact
        self.Dact = Dact
        self.epoch = epoch + 1
        self.trainX = trainX
        ## not missing Matrix로 변환
        self.trainM = 1-1*trainM
        self.testX = testX
        self.testM = 1-1*testM
        self.total_X = np.concatenate((trainX , testX))
        self.total_M = np.concatenate((self.trainM , self.testM))
        self.Train_No , self.Dim = self.total_X.shape
        self.total = self.Dim
        self.p_hint = hint
        self.mb_size = mb_size
        self.alpha = alpha
        ## modeling
        tf.reset_default_graph()
        # 1.1. Data Vector
        self.X = tf.placeholder(tf.float32, shape = [None, self.Dim] , name = "true_data")
        # 1.2. Mask Vector  (미싱 아닌 부분 표시된 것)
        self.M = tf.placeholder(tf.float32, shape = [None, self.Dim] , name = "mask_vector")
        # 1.3. Hint vector
        self.H = tf.placeholder(tf.float32, shape = [None, self.Dim])
        # 1.4. X with missing values
        self.New_X = tf.placeholder(tf.float32, shape = [None, self.Dim] , name = "missing_data")
        self.batch_size = tf.placeholder(tf.int64, name="Batchsize")
        self.dropoutRate = tf.placeholder(tf.float32, name ="dropoutRate")
        
        ## M 은 미싱이 아닌 것!
        ## 미싱이 부분에 진짜를 가짜인 부분에 생성된 것을 
        result = self.generator(self.New_X)
        Logit , G_sample, Result  = result
        
        if self.obj_col == [] :
            for v , col in enumerate(self.Original_Column)  :
                value = tf.slice(G_sample , [0, v ] , [self.batch_size , 1 ] ) # 
                tf.summary.histogram("Input_" + col.replace(" ","_") , value )
        else :
            Result = tf.identity(Result , name = "Arg_G")
            for v , col in enumerate(self.Original_Column)  :
                value = tf.slice(Result , [0, v ] , [self.batch_size , 1 ] ) # 
                tf.summary.histogram("Input_" + col.replace(" ","_") , value )
        Hat_New_X = self.M * self.New_X+ (1-self.M) * G_sample
        
        ## M은 미싱인 부분 
        #imputed = self.New_X * (1-self.M) + G_sample * self.M
        self.Hat_New_X = tf.identity(Hat_New_X , name = "imputed")
        # Discriminator
        D_prob = self.discriminator(Hat_New_X, self.H)
        t_vars = tf.trainable_variables()
        if weight_regularizer > 0 :
            G_L2 = []
            D_L2 = []
            for v in t_vars :
                if re.search('Weight' , v.name) :
                    if re.search("Generator", v.name) :
                        print("G : ", v.name)
                        G_L2.append(tf.nn.l2_loss(v))
                    elif re.search("Discriminator", v.name) :
                        print("D : ", v.name)
                        D_L2.append(tf.nn.l2_loss(v))
            self.Generator_W_l2 =  tf.add_n(G_L2)  * weight_regularizer
            self.Discriminator_W_l2 = tf.add_n(D_L2)  * weight_regularizer
        else :
            self.Generator_W_l2 =  tf.constant(0.0)
            self.Discriminator_W_l2 = tf.constant(0.0)
        for var in t_vars :
            tf.summary.histogram(var.op.name, var)        
        self.D_1 = - self.M * tf.log(D_prob + 1e-8)
        self.D_2 = - (1-self.M) * tf.log(1. - D_prob + 1e-8)
        self.D_3  = tf.reduce_mean(self.D_1 + self.D_2)
        self.D_loss = self.D_3 + self.Discriminator_W_l2
        self.G_loss1 = -tf.reduce_mean( (1-self.M) * tf.log(D_prob + 1e-8))
        ## 미싱이 아닌 부분 -> 미싱 부분
        if self.obj_col == [] : Logit = self.G_final_Act(Logit)
        else : pass
        self.MSE_train_loss =self.CatNumEmb_Loss(Logit , self.X , 1-self.M , seg = "Train")
        self.MSE_train_loss_2 =self.CatNumEmb_Loss(Logit , self.X , self.M , seg = "Test")
        #tf.reduce_sum(tf.reduce_mean(tf.square( self.M * self.New_X - self.M * G_sample  ) , axis = 0))
        #self.CatNumEmb_Loss(Logit , self.X , self.M , seg = "Test")
        self.G_loss =\
        self.G_loss1 + self.alpha * self.MSE_train_loss + self.Generator_W_l2  + self.MSE_train_loss_2
        #+ self.MSE_test_loss
        ## 미싱인 부분에 대해서 잘 맞추는 지 확인하기  
        ## 이런 로스가 아니라 CatNumEmb_Loss 이것과 동일한 로스임 
        #self.MSE_test_loss = self.CatNumEmb_Loss(Logit , self.X , 1-self.M , seg = "Test")
        self.MSE_test_loss =\
        tf.reduce_mean( tf.square( (1-self.M) * self.X - (1-self.M) * G_sample ) )
        
        with tf.variable_scope("Original/Loss"):
            tf.summary.scalar("Total_G_loss", self.G_loss)
            tf.summary.scalar("Not_Missing_Loss", self.MSE_train_loss)
            tf.summary.scalar("D_Loss", self.D_loss)
        
        self.clip_all_weights = tf.get_collection("max_norm")
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
        disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")
        self.global_step = tf.get_variable('global_step',
                                      [], 
                                      initializer=tf.constant_initializer(0), trainable=False)
        self.lr = tf.train.cosine_decay_restarts(lr , self.global_step, 
                                            first_decay_steps= 100  , 
                                            t_mul=1.2 , m_mul=0.95, alpha= 0.5 )
        self.D_solver = RAdamOptimizer(learning_rate=self.lr , 
                                       beta1=0.5, beta2=0.5, 
                                       weight_decay=0.0).minimize(self.D_loss, var_list=disc_vars)
        self.G_solver = tf.train.RMSPropOptimizer(learning_rate=self.lr,).minimize(self.G_loss, var_list=gen_vars)
        self.gen_loss = []
        self.disc_loss = []
        self.train_loss_tr = []
        self.train_loss_te = []
        self.test_loss_tr = []
        self.test_loss_te = []
        comment = "{} \n{}{}{}\n{}".format("="*56 , " "*24, "모델피팅" , " "*24, "="*56 )        
        return print(comment)

    def train(self,) :
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
        if self.curr_iter == 0 :
            it = 0
            ck = "no"
            te_minumum = -1
            tr_minumum = -1
            self.lr_store = []
            config=tf.ConfigProto(log_device_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config = config)
            self.sess.run(tf.global_variables_initializer())
        else :
            print("현재 Epoch : {}까지 진행하였습니다.".format(self.curr_iter))
            num = int(input("추가로 학습을 얼마나 진행하시겠습니까?\n >> "))
            self.epoch += num
            it = self.curr_iter
            ck , tr_minumum , te_minumum = self.ck , self.tr_minumum , self.te_minumum
        
        writer = tf.summary.FileWriter(self.save_model_path)
        writer.add_graph(self.sess.graph )    
        merged_summary = tf.summary.merge_all()
        saver = tf.train.Saver()
        testN , test_dim = self.testX.shape
        test_Z_mb = self.sample_Z(testN , test_dim) 
        test_New_X_mb = self.testM * self.testX + (1-self.testM) * test_Z_mb
        H_mb1 = self.sample_M(testN, self.Dim, 1-self.p_hint)
        test_H_mb = self.testM * H_mb1 + 0.5 * (1-H_mb1)
        count = 0 
        for it in range(it , self.epoch):    
            self.printProgress(it , self.epoch)
            print("Epoch : {} 진행중...".format(it))
            idx = np.random.permutation(self.Train_No)
            XX = self.total_X[idx,:]
            MM = self.total_M[idx,:]  
            batch_iter = int(self.Train_No / self.mb_size)
            Z_mb = self.sample_Z(self.mb_size, self.Dim) 
            Gloss = []
            Dloss = []
            tr_non_missing_loss = []
            tr_missing_loss = []
            self.lr_store.append(self.sess.run(self.lr , feed_dict = {self.global_step : it} ))
            for idx in range(batch_iter) : 
                X_mb = XX[idx*self.mb_size:(idx+1)*self.mb_size]
                M_mb = MM[idx*self.mb_size:(idx+1)*self.mb_size]
                H_mb1 = self.sample_M(self.mb_size, self.Dim, 1-self.p_hint)
                ## 미싱에 힌트를 넣어주는 부분
                H_mb = M_mb * H_mb1 + 0.5 * (1-H_mb1)
                ## 진짜 인 것을 미싱에 넣고 가짜 인 것을 임의로 만들기 
                New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
                FeedDict = {self.M: M_mb, self.New_X: New_X_mb, self.X: X_mb , 
                            self.H: H_mb , self.batch_size : self.mb_size ,
                            self.global_step : it ,
                            self.dropoutRate : 0.05
                           }
                d_step , g_step=5 , 1
                for _ in range(d_step) :
                    _, D_loss_curr , D_3 , Weight_D =\
                     self.sess.run([self.D_solver, self.D_loss , self.D_3 , self.Discriminator_W_l2], 
                         feed_dict = FeedDict)
                
                if self.ck_max_norm : self.sess.run(self.clip_all_weights)
                    
                for _ in range(g_step) :
                    _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr , G_1 , mse , Weight_G=\
                     self.sess.run([
                        self.G_solver, 
                        self.G_loss, 
                        self.MSE_train_loss, self.MSE_test_loss,
                        self.G_loss1 , self.MSE_train_loss , self.Generator_W_l2
                                   ],
                         feed_dict = FeedDict)
                    if self.ck_max_norm : self.sess.run(self.clip_all_weights)
                Dloss.append(D_loss_curr)
                Gloss.append(G_loss_curr)
                tr_non_missing_loss.append(MSE_train_loss_curr)
                tr_missing_loss.append(MSE_test_loss_curr)
            try :
                summary_str = self.sess.run(merged_summary ,  feed_dict = FeedDict)
                writer.add_summary(summary_str , global_step = it)
            except Exception as e :
                self.log.error(e)
                
            D_loss_curr , G_loss_curr = np.mean(Dloss) , np.mean(Gloss)
            Feed = {self.X: self.testX , self.M: self.testM, 
                    self.New_X: test_New_X_mb, self.H: test_H_mb , 
                    self.batch_size :testN , 
                    self.dropoutRate : 0.05}
            tr_non_miss_mean , tr_miss_mean =\
            np.mean(tr_non_missing_loss) , np.mean(tr_missing_loss)
            MSE_test_loss_curr_te = self.sess.run(self.MSE_test_loss,feed_dict = Feed)
            
            self.log.debug("MTrain: {:8.6f}       | MTest  : {:8.6f}".\
                           format(tr_miss_mean ,MSE_test_loss_curr_te))
            self.log.debug("Dloss : {:8.6f}       | Gloss : {:8.6f}".\
                           format(D_loss_curr , G_loss_curr))
            self.log.debug("Dloss : D_3 : {:8.6f} | L2    : {:8.6f}".format(D_3 , Weight_D))
            self.log.debug("Gloss : G_1 : {:8.6f} | MSE   : {:8.6f} | L2 {:.6f}".\
                           format(G_1 , self.alpha*mse , Weight_G))

            try :
                total = np.array(self.test_loss_te) + np.array(self.train_loss_te)
                comp = tr_non_miss_mean + tr_miss_mean # MSE_test_loss_curr
                if it < self.epoch/6 : boolean = min(total[5:]) > comp
                else : boolean = min(self.test_loss_te) > tr_miss_mean # MSE_test_loss_curr_te
                self.log.debug("{:.6f} > {:.6f} ??".\
                               format(min(self.test_loss_te) , tr_miss_mean ))
                ## MSE_test_loss_curr_te
                if boolean == True : 
                    print("{} : {:.6f} -> {:.6f}".
                          format(it , min(total[5:]) , comp))
                    print("Change Missing Min Loss : {:.6f} -> {:.6f}".\
                          format(min(self.test_loss_te) , tr_miss_mean )) # MSE_test_loss_curr_te
                    
                    saver.save(self.sess,  self.save_model_path )
                    ck = it 
                    te_minumum = MSE_test_loss_curr_te
                    tr_minumum = MSE_test_loss_curr
            except Exception as e :
                if it > 9 :
                    print(e)
                pass

            self.gen_loss.append(G_loss_curr)
            self.disc_loss.append(D_loss_curr)
            self.train_loss_te.append(tr_non_miss_mean) # MSE_test_loss_curr
            self.test_loss_te.append(tr_miss_mean) # MSE_test_loss_curr_te
            self.curr_iter = it
            self.ck , self.tr_minumum , self.te_minumum = ck , tr_minumum , te_minumum
            
            if it >= (self.epoch / 5) :
                value = np.mean(total[( it-self.patience) : it])
                diff = value - comp
                if  np.abs(diff)  < self.cut_off :
                    print(diff , self.cut_off)
                    count +=1
                    if count > 10 :
                        print("차이 {} , Cut off {}".format(diff , self.cut_off))
                        break
                    else : pass
                else :pass
            
            if (it % 10 == 0) & (it > 0) :
                clear_output()
                self.vis()
                plt.title("[ Epoch : {} ] GLoss : {:.6f} DLoss : {:.6f}"\
                          .format(it , G_loss_curr , D_loss_curr ) ,
                          fontsize = 20)
                png_file = os.path.join(self.save_model_path , "LossPlot.png")
                plt.savefig(png_file)
                plt.show()    
                if it % 100 == 0 :
                    fig , ax = plt.subplots(figsize = (15,3))
                    plt.subplots_adjust(left = 0.18, bottom = 0.1, right = 0.95 ,
                                        top = 0.95 , hspace = 0, wspace = 0)
                    plt.plot( np.arange(len(self.lr_store)) , self.lr_store) 
                    plt.show()
                    
                try :
                    self.metric_plot(idx = 10)
                except Exception as e : 
                    print(e)
                    pass
        comment = "{} \n      {} \n{}".format("="*20 , "학습완료" , "="*20 )        
        
        return print(comment)
    
    def vis(self , ) :
        fig , ax = plt.subplots(figsize = (15,5))
        plt.subplots_adjust(left = 0.18, bottom = 0.1, right = 0.95 ,
                                        top = 0.95 , hspace = 0, wspace = 0)
        len_ = len( self.gen_loss)
        plt.plot(np.arange(len_) , self.gen_loss, label = "Generator LOSS" , color = "steelblue" )
        plt.plot(np.arange(len_) , self.disc_loss , label = "Discriminator LOSS" , color = "green")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                          fancybox=True, shadow=True, ncol=2)
    
    def loss_plot(self,) :
        self.vis()
        plt.title("GAIN Loss plot", fontsize = 20)
        plt.show()
        
    def metric_plot(self, idx= 50) :
        it = len(self.train_loss_te[idx:])
        fig , axs = plt.subplots(1,2, figsize = (15,5))
        plt.subplots_adjust(left = 0.18, bottom = 0.1, right = 0.95 ,
                                        top = 0.95 , hspace = 0, wspace = 0)
        total = np.array(self.train_loss_te) +  np.array(self.test_loss_te)
        Range = np.arange(idx, it+ idx)
        axs[0].plot(Range , total[idx:] , 
                 linestyle = 'solid', label = "total_loss", color = "magenta")
        axs[0].plot(Range , self.train_loss_te[idx:] , 
                 linestyle = 'solid', label = "train_loss_te", color = "steelblue")
        axs[1].plot(Range , self.test_loss_te[idx:] , 
                 linestyle = 'solid', label = "test_loss_te", color = "green")
        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                      fancybox=True, shadow=True, ncol=4)
        axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                      fancybox=True, shadow=True, ncol=4)
        plt.suptitle("[ Epoch : {} ] train min : {:.5f}, test min : {:.5f}"\
                  .format(self.ck , self.tr_minumum , self.te_minumum ) ,
                  fontsize = 20)
        axs[0].axvline(self.ck, color='k', linestyle='dashed')
        axs[1].axvline(self.ck, color='k', linestyle='dashed')
        png_file = os.path.join(self.save_model_path , "MetricPlot.png")
        plt.savefig(png_file)
        plt.show()
    
    ### missing 부분만 비교하기!






