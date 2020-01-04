import numpy as np
from IPython.display import clear_output
import pandas as pd
import tensorflow as tf
from tqdm import tqdm , tqdm_notebook
import numpy as np
import re , os
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings , logging
import dill
import warnings
import sys
sys.path.append('/home/advice/Python/SR/Custom/')
from RAdam import RAdamOptimizer
warnings.filterwarnings('ignore')
import shutil

def ck_dir(dir_) :
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    else :
        shutil.rmtree(dir_)
        os.makedirs(dir_)


from Common import Com , Embedding , tf_mish

class Missing_AE(Com) :
    """
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
    def __init__(self, enc_dim , dec_dim , missing_info , 
                 save_info_path = None ,
                 save_model_path = './CAT_AE/Model' , gpu = "0" ) :
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        ########################################
        try :
            print("{}에 missing info 저장".format(save_info_path))
            with open(save_info_path, 'wb') as f:
                dill.dump(missing_info , f)
        except Exception as e :
            print("Error : " ,  e)
            raise
        self.missing_info = missing_info
        ck_dir(save_model_path)
        self.save_model_path  = save_model_path 
        ########################################
        self.obj_range = missing_info["obj_range"]
        self.obj_info = missing_info["obj_info"]
        self.ce_one_hot = missing_info["ce_encoder"]
        self.columns= missing_info["columns"]
        self.scaler = missing_info["scaler"]
        self.num_col = missing_info["num_col"]
        self.ord_col = missing_info["ord_col"]
        self.obj_col = missing_info["obj_col"]
        self.na_add_col = missing_info["ce_replace"]
        self.cond = missing_info["cat_num_idx_info"]
        self.Original_Column = missing_info["original_column"]
        self.curr_iter = 0
        self.scale_range = missing_info["scaler"].feature_range
        if self.scale_range == (0,1) :
            self.G_final_Act = tf.nn.sigmoid
        elif self.scale_range == (-1,1) :
            self.G_final_Act = tf.nn.tanh
        elif self.scale_range == (-2,2) :
            self.G_final_Act = tf.nn.tanh
        os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)

    def encoder(self , missing_X) :
        all_h = self.enc_dim
        
        
        activation = self.Enc_act
        init = self.W_Init(activation) 
        if self.obj_col == [] : pass
        else :
            missing_X = Embedding(missing_X , self.cond , self.total , self.batch_size)
        
        print("Input Shape : {}".format(missing_X.shape))
        print("="*56) 
        with tf.variable_scope("AutoEncoder/Encoder"):
            for idx , h_dim in enumerate(self.enc_dim) :
                if idx == 0 :
                    ENC_Weight = tf.get_variable("EncWeight_" + str(idx) , dtype = tf.float32 , 
                                 shape = [missing_X.shape[1] , h_dim] , initializer = init)
                    if self.ck_SN : ENC_Weight = self.spectral_norm(ENC_Weight, 
                                                                    name ="Enc_SN_Weight"+str(idx))
                    else : pass
                    ENC_bias = tf.get_variable("EncBias_" + str(idx) , shape = [h_dim] , dtype = tf.float32 , 
                                         initializer = tf.constant_initializer(0.0))
                    Layer = self.Enc_act(tf.matmul( missing_X , ENC_Weight) + ENC_bias)
                    if self.Enc_act  == tf.nn.selu    :
                        Layer = tf.contrib.nn.alpha_dropout(Layer , self.dropoutRate ) 
        
                else :
                    ENC_Weight = tf.get_variable("EncWeight_" + str(idx) , dtype = tf.float32 , 
                                 shape = [all_h[idx-1] , all_h[idx]] , initializer = init)
                    if self.ck_SN : ENC_Weight = self.spectral_norm(ENC_Weight, 
                                                                    name ="Enc_SN_Weight"+str(idx))
                    else : pass
                    ENC_bias = tf.get_variable("EncBias_" + str(idx) , 
                                               shape = [all_h[idx]] , dtype = tf.float32 , 
                                         initializer = tf.constant_initializer(0.0))
                    Layer = tf.matmul( Layer , ENC_Weight) + ENC_bias
                    if len(self.enc_dim) == idx+1 :pass
                    else : 
                        Layer = self.Enc_act(Layer)
                        if self.Enc_act== tf.nn.selu  :
                            Layer = tf.contrib.nn.alpha_dropout(Layer , self.dropoutRate ) 
                        
        return Layer

    def decoder(self, X) :
        all_h = self.dec_dim
        activation = self.Dec_act
        init = self.W_Init(activation) 
        with tf.variable_scope("AutoEncoder/Decoder"):
            for idx , h_dim in enumerate(self.dec_dim) :
                if idx == 0 :
                    DEC_Weight = tf.get_variable("DecWeight_" + str(idx) , dtype = tf.float32 , 
                                 shape = [self.enc_dim[-1] , h_dim] , initializer = init)
                    if self.ck_SN : DEC_Weight = self.spectral_norm(DEC_Weight,
                                                                    name ="Dec_SN_Weight"+str(idx))
                    else : pass
                    DEC_bias = tf.get_variable("DecBias_" + str(idx) , 
                                               shape = [h_dim] , dtype = tf.float32 , 
                                               initializer = tf.constant_initializer(0.0))
                    Layer = self.Dec_act(tf.matmul( X , DEC_Weight) + DEC_bias)
                    if self.Dec_act == tf.nn.selu   :
                        Layer = tf.contrib.nn.alpha_dropout(Layer , self.dropoutRate ) 
        
                else :
                    DEC_Weight = tf.get_variable("DecWeight_" + str(idx) , dtype = tf.float32 , 
                                 shape = [all_h[idx-1] , all_h[idx] ] , initializer = init)
                    if self.ck_SN : DEC_Weight = self.spectral_norm(DEC_Weight, 
                                                                    name ="Dec_SN_Weight"+str(idx))
                    else : pass
                    DEC_bias = tf.get_variable("DecBias_" + str(idx) , 
                                               shape = [all_h[idx]] , dtype = tf.float32 , 
                                               initializer = tf.constant_initializer(0.0))
                    Layer = self.Dec_act(tf.matmul( Layer , DEC_Weight) + DEC_bias)
                    if self.Dec_act == tf.nn.selu   :
                        Layer = tf.contrib.nn.alpha_dropout(Layer , self.dropoutRate ) 
            
            activation = self.G_final_Act
            init = self.W_Init(activation) 
            DEC_Weight = tf.get_variable("DecWeight_Final" , dtype = tf.float32 , 
                                 shape = [self.dec_dim[-1] , self.total] , initializer = init)
            if self.ck_SN : 
                DEC_Weight = self.spectral_norm(DEC_Weight, 
                                                name ="Dec_SN_Weight_Final")
            DEC_bias = tf.get_variable("DecBias_Final" + str(idx) , 
                                       shape = [self.total] , 
                                       dtype = tf.float32 , 
                                       initializer = tf.constant_initializer(0.0))
            Logit = tf.matmul( Layer , DEC_Weight) + DEC_bias
            
            if self.obj_col == [] :
                output = self.G_final_Act(Logit)
                argmax = None
                for v , col in enumerate(self.Original_Column)  :
                    value = tf.slice(output , [0, v ] , [self.batch_size , 1 ] ) # 
                    tf.summary.histogram("Input_" + col.replace(" ","_") , value )    
            else :
                output , argmax = self.CatNumEmb(Logit)

        return output , argmax

    def fit(self, trainX , trainM ,testX ,testM   ,
            mb_size , alpha = 3 , epoch = 1000,
            Enc_act = tf.nn.selu , Dec_act = tf.nn.selu , 
            weight_regularizer = 0.005 , SN = False ,
            patience = 10, cutoff = 0.001 , lr = 0.001 ,
            MetricLoss = False , 
           ) :
        """
        trainM : missing matrix
        testM : missing matrix
        """
        tf.reset_default_graph()
        select_w_init = np.random.randint(0, 2, size=1 )[0]
        seed_n = np.random.randint(1, 1000, size=1 )[0]
        self.patience = patience
        self.cut_off = cutoff
        self.alpha = alpha
        self.relu_w_init = [tf.keras.initializers.he_uniform(seed = seed_n) ,
                       tf.keras.initializers.he_normal(seed = seed_n)][select_w_init]
        self.tanh_w_init = [tf.keras.initializers.glorot_normal(seed = seed_n) ,
                       tf.keras.initializers.glorot_uniform(seed = seed_n)][select_w_init]
        self.s_elu_w_init = [tf.keras.initializers.lecun_normal(seed = seed_n) ,
                       tf.keras.initializers.lecun_uniform(seed = seed_n)][select_w_init]
        self.nomal_w_init = tf.keras.initializers.truncated_normal(seed = seed_n)
        self.ck_SN = SN
        self.Enc_act = Enc_act
        self.Dec_act = Dec_act
        self.epoch = epoch + 1
        self.trainX = trainX
        self.trainM = 1-1*trainM
        self.testX = testX
        self.testM = 1-1*testM
        self.total_X = np.concatenate((trainX , testX))
        self.total_M = np.concatenate((self.trainM , self.testM))
        self.Train_No , self.Dim = self.total_X.shape
        self.mb_size = mb_size
        self.total = self.Dim
        self.X = tf.placeholder(tf.float32, shape = [None, self.Dim] , name = "true_data")
        self.New_X = tf.placeholder(tf.float32, shape = [None, self.Dim] , name = "missing_data")
        self.M = tf.placeholder(tf.float32, shape = [None, self.Dim] , name = "mask_vector")
        self.batch_size = tf.placeholder(tf.int64, name="Batchsize")
        self.dropoutRate = tf.placeholder(tf.float32, name ="dropoutRate")
        enc = self.encoder(self.New_X)
        self.enc = enc
        new_x , arg = self.decoder(enc)
        self.new_x = new_x
        
        
        ## M 은 미싱이 아닌 것!
        Hat_New_X = self.M * self.X  + (1-self.M) * new_x
        self.Hat_New_X = tf.identity(Hat_New_X , name = "imputed")
        mean, var = tf.nn.moments(self.X, axes=[0, 1])
        self.MSE_true_loss =\
        tf.reduce_sum(tf.reduce_mean(tf.square( self.M * new_x  - self.M * self.X  ) , axis = 0))
        #tf.reduce_mean(tf.square( self.M * new_x  - self.M * self.X  ))
        #/ tf.reduce_mean(self.M)
        self.MSE_missing_loss =\
        tf.reduce_sum(
            tf.reduce_mean( 
                tf.square( (1-self.M)*(new_x) - (1-self.M)*(self.X) ) , 
            axis = 0)
        )
        if MetricLoss :
            self.metric_loss = self.CatNumEmb_Loss(new_x , self.X , self.M, seg= "Train")
        else : 
            self.metric_loss = tf.constant(0.0)
        self.MSE_lOSS = tf.reduce_mean(tf.square(new_x - self.X ))

        t_vars = tf.trainable_variables()
        if weight_regularizer > 0.0 :
            Weight_Reg = []
            for v in t_vars :
                if re.search('Weight' , v.name) :
                    print("Weight : ", v.name)
                    Weight_Reg.append(tf.nn.l2_loss(v))
            W_l2 = tf.add_n(Weight_Reg)  * weight_regularizer
        else : W_l2 = tf.constant(0.0)
        for var in t_vars : 
            tf.summary.histogram(var.op.name, var)
        tf.summary.scalar("True_Loss", self.MSE_true_loss)
        tf.summary.scalar("MSE_Loss", self.alpha * self.MSE_missing_loss)
        enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="AutoEncoder/Decoder")
        dec_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="AutoEncoder/Encoder")
        self.Loss =\
        self.MSE_true_loss + self.alpha * self.MSE_missing_loss  + W_l2  + self.metric_loss
        self.global_step = tf.get_variable('global_step',[], 
                                           initializer=tf.constant_initializer(0), trainable=False)
        
        # http://www.deeplearningessentials.science/hyperParameters/
        # https://github.com/yg-sbm/deepest_quest1
        self.lr = tf.train.cosine_decay_restarts(lr , self.global_step, 
                                            first_decay_steps= 50 ,  
                                            t_mul=1.2 , m_mul=0.999, alpha= 0.5 ) 
        self.solver = RAdamOptimizer(learning_rate= self.lr , 
                                     beta1=0.5, beta2=0.5, weight_decay=0.0).\
        minimize(self.Loss, var_list=enc_vars + dec_vars)
        self.train_loss_tr = []
        self.train_loss_te = []
        self.test_loss_tr = []
        self.test_loss_te = []
        comment = "{} \n{}{}{}\n{}".format("="*56 , " "*24, "모델피팅" , " "*24, "="*56 )        
        return print(comment)
        
    def train(self,) :
        if self.curr_iter == 0 :
            it = 0
            ck = "no"
            te_minumum = 999
            tr_minumum = 999
            self.lr_store = []
            config=tf.ConfigProto(log_device_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config = config)
            self.sess.run(tf.global_variables_initializer())
            self.writer = tf.summary.FileWriter(self.save_model_path)
            self.writer.add_graph(self.sess.graph )
        else :
            print("현재 Epoch : {}까지 진행하였습니다.".format(self.curr_iter))
            num = int(input("추가로 학습을 얼마나 진행하시겠습니까?\n >> "))
            self.epoch += num
            it = self.curr_iter
            ck , tr_minumum , te_minumum = self.ck , self.tr_minumum , self.te_minumum
        
        
        saver = tf.train.Saver()
        ######################################
        self.log = logging.getLogger("")
        if (self.log.hasHandlers()): self.log.handlers.clear()
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = True
        formatter = logging.Formatter("%(asctime)s | %(message)s",
                                      "%Y-%m-%d %H:%M:%S")
        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(logging.DEBUG)
        streamHandler.setFormatter(formatter)
        self.log.addHandler(streamHandler)
        ######################################
        count = 0 
        for it in range(it , self.epoch) :    
            #%% Inputs
            self.printProgress(it , self.epoch)
            print("Epoch : {} 진행중...".format(it))
            idx = np.random.permutation(self.Train_No)
            XX = self.total_X[idx,:]
            MM = self.total_M[idx,:]  
            batch_iter = int(self.Train_No / self.mb_size)
            Z_mb = self.sample_Z(self.mb_size, self.Dim) 
            trloss = []
            teloss = []
            self.lr_store.append(self.sess.run(self.lr , feed_dict = {self.global_step : it} ))
            for idx in range(batch_iter) :
                X_mb = XX[idx*self.mb_size:(idx+1)*self.mb_size]
                M_mb = MM[idx*self.mb_size:(idx+1)*self.mb_size]
                New_X_mb = M_mb * X_mb + (1-M_mb)  * Z_mb  # Missing Data Introduce
                Feed = {
                    self.M: M_mb, 
                    self.New_X: New_X_mb, 
                    self.X : X_mb , 
                    self.batch_size : self.mb_size ,
                    self.global_step : it ,
                    self.dropoutRate : 0.05
                       }
                _, train_loss_curr , MSE_true_loss_curr , MSE_test_loss_curr  =\
                self.sess.run([self.solver, self.Loss ,
                               self.MSE_true_loss , self.MSE_missing_loss], 
                     feed_dict = Feed)
                trloss.append(MSE_true_loss_curr)
                teloss.append(MSE_test_loss_curr)
                
            MSE_true_loss_curr = np.mean(trloss)
            MSE_test_loss_curr = np.mean(teloss)
            self.log.debug("Epoch {:4} True : {:9.6f} | Missing            : {:9.6f} ".\
                           format(it , MSE_true_loss_curr , MSE_test_loss_curr))
            test_N , test_dim = self.testX.shape
            test_Z_mb = self.sample_Z(test_N , test_dim) 
            test_New_X_mb = self.testM * self.testX + (1-self.testM)  * test_Z_mb
            testFeed = {self.M: self.testM, self.New_X: test_New_X_mb, 
                        self.X : self.testX , self.batch_size : test_N ,
                        self.dropoutRate : 0.05
                       }
            test_loss_curr , test_true , test_missing =\
            self.sess.run([self.Loss , 
                           self.MSE_true_loss ,
                           self.MSE_missing_loss], 
                          feed_dict =testFeed)
            self.log.debug("Train True Loss : {:9.6f} | Train Missing Loss : {:9.6f}".\
                                   format(MSE_true_loss_curr , MSE_test_loss_curr))
            self.log.debug("Test  True Loss : {:9.6f} | Test  Missing Loss : {:9.6f}".\
                           format(test_true , test_missing))
            try :
                Comp = MSE_true_loss_curr + MSE_test_loss_curr
                #self.test_loss_te
                Train_Total_Loss = np.array(self.train_loss_tr) + np.array(self.train_loss_te)
                if it < self.epoch/6 : boolean = min(Train_Total_Loss[5:]) > Comp
                else : boolean = min(self.train_loss_te) > MSE_test_loss_curr
                self.log.debug("Total   {:.6f} > {:.6f} ??".\
                               format(min(Train_Total_Loss) , Comp))
                self.log.debug("Missing {:.6f} > {:.6f} ??".\
                               format(min(self.train_loss_te) ,MSE_test_loss_curr))
                if boolean == True :
                    self.log.debug("Yes! [{}] Change : {:.6f} ->> {:.6f}".\
                                   format(it ,min(Train_Total_Loss) , Comp))
                    
                    count = 0
                    saver.save(self.sess,  
                               self.save_model_path )
                    ck = it 
                    te_minumum = test_missing
                    tr_minumum = MSE_test_loss_curr
            except ValueError :
                pass
            except Exception as e  :
                print(e)
                raise
            self.train_loss_tr.append(MSE_true_loss_curr)
            self.train_loss_te.append(MSE_test_loss_curr)
            self.test_loss_tr.append(test_true)
            self.test_loss_te.append(test_missing)
            
            if it >= (self.epoch / 5) :
                #diff = np.mean(Train_Total_Loss[ (it-self.patience) : it]) - Comp
                diff =\
                np.mean(np.array(self.train_loss_te)[ (it-self.patience) : it]) - MSE_test_loss_curr
                if  np.abs(diff)  < self.cut_off :
                    count +=1
                    if count > 10 :
                        print("차이 {} , Cut off {}".format(diff , self.cut_off))
                        break
                    else : pass
                else :pass

            
            #%% Intermediate Losses
            if (it % 10 == 0) & (it > 0) :
                clear_output()
                try :
                    self.loss_plot()                
                    plt.suptitle("[ Epoch : {} ] train min : {:.5f} test min : {:.5f}"\
                              .format(ck , tr_minumum , te_minumum ) ,
                              fontsize = 20)
                    plt.axvline(ck, color='k', linestyle='dashed')
                    png_file = os.path.join(self.save_model_path , "MetricPlot.png")
                    plt.savefig(png_file)
                    plt.show()
                    if it % 100 == 0 :
                        fig , ax = plt.subplots(figsize = (15,3))
                        plt.plot( np.arange(len(self.lr_store)) , self.lr_store) 
                        plt.show()
                except :
                    pass
            self.curr_iter = it
            self.ck , self.tr_minumum , self.te_minumum = ck , tr_minumum , te_minumum
        comment = "{} \n      {} \n{}".format("="*20 , "학습완료" , "="*20 )        
        return print(comment)
                
    
    
    def loss_plot(self,) :
        fig , axs = plt.subplots(1,2, figsize = (15,5))
        plt.subplots_adjust(left = 0.18, bottom = 0.1, right = 0.95 ,
                                        top = 0.95 , hspace = 0, wspace = 0)
        idx = 100
        len_ = len(np.array(self.train_loss_tr[idx:] ))
#         plt.plot(np.arange(len_) , 
#                  np.array(self.train_loss_tr) +  np.array(self.train_loss_te ), 
#                  linestyle = 'solid' , label = "train_loss_total" , color = "steelblue")
        axs[0].plot(np.arange(idx , len_+idx) , self.train_loss_tr[idx:] , 
                 linestyle = 'dashed' , label = "train_loss_tr", color = "steelblue")
        
#         plt.plot(np.arange(len_) , np.array(self.test_loss_tr) +  np.array(self.test_loss_te ),
#                  linestyle = 'solid' , label = "test_loss_total", color = "green")
        axs[0].plot(np.arange(idx , len_+idx) , self.test_loss_tr[idx:] , 
                 linestyle = 'dashed' , label = "test_loss_tr", color = "green")
        axs[1].plot(np.arange(idx , len_+idx) , self.train_loss_te[idx:] , 
                 linestyle = 'dotted', label = "train_loss_te", color = "steelblue")
        axs[1].plot(np.arange(idx , len_+idx) , self.test_loss_te[idx:] , 
                 linestyle = 'dotted', label = "test_loss_te", color = "green")
        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=True, ncol=2)
        axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=True, ncol=2)
        
    

#     def object_comparision(self , Real , Imputed ) :
#         """
#         > Return : object missing comparision 
#         """
#         obj_col = self.obj_col
#         ck = ~(Real[obj_col] == Imputed[obj_col])
#         real_missing = self.missing_info["ori_missing_matrix"]
#         missing_n = pd.DataFrame(real_missing, 
#                                  columns = Real.columns.tolist())[obj_col].sum(axis = 0)
#         ratio = ck.sum(axis = 0) / missing_n
#         ratio.plot.barh(title= "Object Columns Missing Predict Ratio")
#         plt.show()
#         result = pd.concat([ck.sum(axis = 0) , missing_n], axis = 1 ).\
#         rename( columns = {0 :"N_not_correct" , 1 : "N_missing"})
#         result["Ratio"] = result["N_not_correct"] / result["N_missing"] 
#         return result