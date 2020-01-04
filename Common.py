import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
import re , os
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings
import dill
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')
import sys 
from sklearn.model_selection import KFold

def tf_mish(x) :
    return x * tf.nn.tanh(tf.nn.softplus(x))

def focal_loss_sigmoid(labels,logits,alpha=0.25,gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-labels*(1-alpha)*((1-y_pred)*gamma)*tf.log(y_pred)-\
      (1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)
    return L

def focal_loss_softmax(labels,logits , gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.softmax(logits,dim=-1) # [batch_size,num_classes]
    labels=tf.one_hot(labels,depth=y_pred.shape[1])
    L=-labels*((1-y_pred)**gamma)*tf.log(y_pred)
    L=tf.reduce_sum(L,axis=1)
    return L

def Embedding(X, condition , total_dim , batch_size ) :
    cond = condition + [total_dim]
    inputs = []
    for idx in range(len(condition)) :
        diff = cond[idx+1] - cond[idx] 
        split = tf.slice(X , [0 , cond[idx] ] ,
                         [batch_size , cond[idx+1]-cond[idx]] ) # 
        if diff < 6 :
            Numeric = split
            inputs.append(Numeric)
        else :
            embeddings = tf.Variable(tf.truncated_normal([diff, int(diff/2)], 
                                                         stddev = 2/diff ))       
            split = tf.argmax(split , axis = 1)
            embedding = tf.nn.embedding_lookup(embeddings, split)
            inputs.append(embedding)
    concatenated_layer = tf.concat(inputs, axis=1, name='concatenate')
    print("Onehot Shape : [{}] --> Embedding Shape : [{}] ".\
          format(total_dim , concatenated_layer.get_shape().as_list()[1] ))
    return concatenated_layer


class Com :
    def W_Init(self , act ) :
        if act in [tf.nn.leaky_relu, tf.nn.relu] :  init = self.relu_w_init
        elif act in [tf.nn.tanh , tf.nn.softmax , tf.nn.sigmoid] :  init = self.tanh_w_init
        elif act in [tf.nn.selu , tf.nn.elu] :      init = self.s_elu_w_init
        else : init = self.s_elu_w_init
        return init
    
    def printProgress (self , iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 50): 
        formatStr = "{0:." + str(decimals) + "f}" 
        percent = formatStr.format(100 * (iteration / float(total))) 
        filledLength = int(round(barLength * iteration / float(total))) 
        bar = '#' * filledLength + '-' * (barLength - filledLength) 
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)), 
        if iteration == total: 
            sys.stdout.write('\n') 
        sys.stdout.flush()
    
    def nrmse(self , true , imp ) : 
        """normalized root mse"""
        n = np.mean((true - imp)**2 , axis = 0 )
        d = np.var(true ,axis = 0) + 1e-2
        return np.mean(n/d)
    
    def rmse(self , true , imp) :
        """root mse"""
        n = np.mean(((imp - true ))**2 , axis = 0)
        n = np.sqrt(n)
        return np.mean(n)
    
    def metric(self , Real , Imputed , Missing , standardize , kf  ,  metric_type = "rmse") :
        num_ = self.num_col
        ord_ = self.ord_col
        scaler = self.missing_info["scaler"]
        ss = ord_ + num_
        R = deepcopy(Real[ss])
        I = deepcopy(Imputed[ss])
        M = pd.DataFrame(Missing, columns = self.missing_info["columns"])[ss].values
        if standardize :
            R[ss] = scaler.transform(R[ss])
            I[ss] = scaler.transform(I[ss])
        #D = ((R-I))**2 # *M
        
        if kf :
            result = []
            for train_index, test_index in kf :
                #cv = np.sqrt(np.mean(D.iloc[test_index,:].mean()))
                cv_R = R.iloc[test_index,:]
                cv_I = I.iloc[test_index,:]
                if metric_type == "rmse" : cv = self.rmse(cv_R , cv_I)
                elif metric_type == "nrmse" : cv = self.nrmse(cv_R , cv_I)
                else : raise
                result.append(cv)
        else :
            if metric_type == "rmse" : result = self.rmse(R , I)
            elif metric_type == "nrmse" : result = self.nrmse(R , I)
            else : raise
        ## np.sqrt( np.mean((M * I.values  - M * R.values )**2 ) / np.mean(M) )
        return result
        
    def object_comparision(self , Real , Imputed , target = None ) :
        """
        > Return : object missing comparision 
        """
        obj_col = self.obj_col
        ck = ~(Real[obj_col] == Imputed[obj_col])
        real_missing = self.missing_info["ori_missing_matrix"]
        missing_n = pd.DataFrame(real_missing, 
                                 columns = Real.columns.tolist())[obj_col].sum(axis = 0)
        ratio = ck.sum(axis = 0) / missing_n
        if target is not None :
            ratio = ratio.drop(index = [target])
        ratio.plot.barh(title= "Object Columns Missing Predict Ratio")
        png_file = os.path.join(self.save_model_path , "object_comparision.png")
        plt.savefig(png_file)
        plt.show()
        result = pd.concat([ck.sum(axis = 0) , missing_n], axis = 1 ).\
        rename( columns = {0 :"N_not_correct" , 1 : "N_missing"})
        ## Proportion of Falsely Classified entries
        result["PFC"] = result["N_not_correct"] / result["N_missing"] 
        if target is not None :
            result = result.drop(index = [target])
        save_file = os.path.join(self.save_model_path , "object_comparision_result.csv")
        result.to_csv(save_file , index = True)
        return result
    
    def sample_Z(self , m, n):
        return np.random.uniform(0., 1, size = [m, n])    

    def sample_idx(self , m, n):
        A = np.random.permutation(m)
        idx = A[:n]
        return idx

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape = size, stddev = xavier_stddev)
    
    def CatNumEmb(self , out  ) :
        condition = self.cond
        cond = condition + [self.total]
        inputs = []
        args   = []
        for idx in range(len(condition)) :
            diff = cond[idx+1] - cond[idx] 
            split = tf.slice(out , 
                             [0 , cond[idx] ] ,
                             [self.batch_size , cond[idx+1]-cond[idx]] ) # 
            if diff == 1 :
                first = self.G_final_Act(split)
                arg = self.G_final_Act(split)
                inputs.append(first)
                args.append(arg)
            else :
                first = tf.nn.softmax(split)
                arg = tf.expand_dims(tf.argmax(first , axis = 1 ) ,axis = 1 )
                arg = tf.cast(arg , dtype = tf.float32)
                inputs.append(first)
                args.append(arg)
        Input = tf.concat(inputs, axis=1, name='Inputs')
        Arg = tf.concat(args, axis=1, name='Args')
        return Input , Arg
    
    def impute(self, X , matrix , trans = True) :
        """
        matrix : missing matrix (1,0)
        """
        N , DIM = X.shape
        mb = self.sample_Z(N , DIM) 
        X = X.astype(float)
        where = np.isnan(X)
        X[where] = 100
        matrix  = 1-matrix
        New_X_mb = matrix * X  + (1-matrix) * mb

        self.Imputed = self.sess.run(self.Hat_New_X, 
                           feed_dict =\
                           {self.M: matrix , self.New_X: New_X_mb, 
                            self.X : X , self.batch_size : N })
        if trans :
            output = self.transform(self.Imputed)
        else :
            output = self.Imputed
        return output

    def max_norm_regularizer(self , threshold, axes=1, name="max_norm",collection="max_norm"):
         def max_norm(weights):
            clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
            clip_weights = tf.assign(weights, clipped, name=name)
            tf.add_to_collection(collection, clip_weights)
            return None # there is no regularization loss term
         return max_norm

    def spectral_norm(self, w, iteration=2 , name = None):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        u = tf.get_variable(name , [1, w_shape[-1]], 
                            initializer=tf.random_normal_initializer(), 
                            trainable=False)
        u_hat = u
        v_hat = None
        for i in range(iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)
        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)
        return w_norm 

    def transform(self , imp , num_inv_scaler = True) :
        output = pd.DataFrame(imp , columns = self.columns)
        self.before_trans = output
        for col , dict_ in self.obj_info.items() :
            start = dict_["start_idx"]
            n = dict_["n"]
            output.iloc[: , start:(start+n)] =\
            output.iloc[: , start:(start+n)].apply(lambda x : 1*( max(x) == x ) ,  axis = 1)
        for col in self.na_add_col :
            output[col] = np.nan
        imputed_output = self.ce_one_hot.inverse_transform(output)
        cols = self.ord_col + self.num_col
        if num_inv_scaler == True :
            imputed_output[cols] =  self.scaler.inverse_transform(imputed_output[cols])
        return imputed_output
    
    def load_impute(self , save_file ,file , Data , Missing , trans = True) :
        """
        Missing : missing matrix (1,0)
        """
        N , DIM = Data.shape
        mb = self.sample_Z(N , DIM) 
        Data = Data.astype(float)
        where = np.isnan(Data)
        Data[where] = 100
        Missing = 1 - Missing
        New_X_Miss = Missing * Data  + (1-Missing) * mb
        tf.reset_default_graph()
        config=tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config= config)
        saver = tf.train.import_meta_graph(save_file)
        saver.restore(sess, tf.train.latest_checkpoint(file))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('true_data:0') 
        M = graph.get_tensor_by_name('mask_vector:0')
        New_X = graph.get_tensor_by_name('missing_data:0')
        Hat_New_X = graph.get_tensor_by_name('imputed:0')
        batchsize = graph.get_tensor_by_name('Batchsize:0')
        try :
            dropoutrate = graph.get_tensor_by_name('dropoutRate:0')
        except :
            pass
        dic = {X: Data, M: Missing, 
               New_X: New_X_Miss ,
               batchsize : len(Data),
               dropoutrate : 0.1
              }
        imputed = sess.run(Hat_New_X, feed_dict = dic)
        sess.close()
        if trans :
            output = self.transform(imputed)
        else :
            output = imputed
        return output
    
    def CatNumEmb_Loss(self ,  Gene , Real, M  , seg ) :
        condition = self.cond
        origin_col = self.Original_Column
        LOSS = 0
        cond = condition + [ self.total]
        with tf.variable_scope("Columns/Loss"):
            if len(condition) == self.total :
                sumLoss = tf.reduce_mean(M * tf.square( Real - Gene ))
                sumLoss /= tf.clip_by_value(tf.reduce_mean(M),1e-8,100)
                LOSS = tf.clip_by_value(sumLoss,0.0,100)
                if seg == "Train" :
                    print("======================")
                    print(" Only Numeric Columns ")
                    print("======================")
                tf.summary.scalar("TOTAL_MSE", LOSS )    
            else :
                for idx in range(len(condition)) :
    #                 try :
                    diff = cond[idx+1] - cond[idx] 
                    Gsplit = tf.slice(Gene , [0 , cond[idx] ] ,
                                     [self.batch_size , diff] ) # 
                    Rsplit = tf.slice(Real , [0 , cond[idx] ] ,
                                     [self.batch_size , diff] ) # 
                    Msplit = tf.slice(M , [0 , cond[idx] ] ,
                                     [self.batch_size , diff] ) # 
                    if diff == 1 :
                        type_ = "mse"
                        #Missing = 1-Msplit
                        Missing = Msplit
                        fake = self.G_final_Act(Gsplit)
                        real = Rsplit
                        sumLoss = tf.reduce_mean(Missing * tf.square( real - fake ))
                        sumLoss /= tf.clip_by_value(tf.reduce_mean(Missing),1e-8,100)
                        sumLoss = tf.clip_by_value(sumLoss,0.0,100)
                    
                    elif diff == 2 :
                        type_ = "binary"
                        #Missing = 1-Msplit
                        Missing = Msplit
                        one_hot = tf.one_hot(tf.argmax( Rsplit , 1 ), depth = diff )
                        """ 
                        https://alexisalulema.com/2017/12/15/classification-loss-functions-part-ii/
                        """
                        weights = self.missing_info["obj_weight_info"][idx]
                        weight = tf.constant( [weights[0]/weights[1]] )
                        WCE = tf.nn.weighted_cross_entropy_with_logits(targets = one_hot ,
                                                                       logits = Gsplit ,pos_weight =  weight)
                        #CE = tf.nn.sigmoid_cross_entropy_with_logits(logits = Gsplit , labels = one_hot )
                        sumLoss =tf.reduce_mean(Missing * WCE )
                        sumLoss /= tf.clip_by_value(tf.reduce_mean(Missing), 1e-8 , 1000)
                        sumLoss = tf.clip_by_value(sumLoss,0.0,100)
                    else :
                        type_ = "multiclass"
                        #Missing = 1-Msplit
                        Missing = Msplit
                        one_hot = tf.one_hot(tf.argmax(Rsplit , 1), depth = diff )
                        Missing = tf.reduce_mean(Missing,axis = 1 )
                        ## softmax 
#                         sumLoss =\
#                         tf.reduce_mean(Missing * tf.nn.softmax_cross_entropy_with_logits(
#                             logits = Gsplit , labels = one_hot))
                        ## https://code-examples.net/ko/q/2a7f0a5
                        ## sparse softmax
                        weights = self.missing_info["obj_weight_info"][idx]
                        class_weights = tf.constant(weights)
                        labels = tf.argmax(Rsplit, axis = 1 )
                        weights = tf.gather(class_weights,  labels)
                        SparseCE = tf.losses.sparse_softmax_cross_entropy(labels, Gsplit , weights)
                        sumLoss =tf.reduce_mean(Missing * SparseCE)
                        ## focal softmax
#                         labels = tf.argmax(Rsplit, axis = 1 )
#                         FocalSoftmax =focal_loss_softmax(labels=labels , logits=Gsplit)
#                         sumLoss =tf.reduce_sum(Missing * FocalSoftmax)
                        
                        sumLoss /= tf.clip_by_value(tf.reduce_mean(Missing),1e-8,100)
                        sumLoss = tf.clip_by_value(sumLoss,0.0,100)
                    colLoss = sumLoss  
                    if seg == "Train" :
                        print("Columns : {:25} | {:10} | [{}]".format(origin_col[idx] , type_ , diff))
                    tf.summary.scalar(str(diff)+"_"+origin_col[idx].replace(" ","_") , colLoss )    
                    LOSS += colLoss
        return LOSS