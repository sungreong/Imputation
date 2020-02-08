import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
import re, os
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings
import dill
from copy import deepcopy
import warnings

warnings.filterwarnings('ignore')
import sys
from sklearn.model_selection import KFold

table = pd.DataFrame


def tf_mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))


def gainloss(M, G, R, weights):
    diff = G.get_shape().as_list()[1]
    if diff == 1:
        sumLoss = mseloss(M, G, R)
    elif diff == 2:
        sumLoss = binaryloss(M, G, R, weights)
    else:
        sumLoss = multiclassloss(M, G, R, weights)
    return sumLoss


def binaryloss(M, G, R, weights):
    diff = G.get_shape().as_list()[1]
    one_hot = tf.one_hot(tf.argmax(R, 1), depth=diff)
    weight = tf.constant([weights[1]])
    WCE = (
        tf.nn.weighted_cross_entropy_with_logits(targets=one_hot, logits=G,
                                                 pos_weight=weight))
    sumLoss = tf.reduce_mean(M * WCE)
    sumLoss /= tf.clip_by_value(tf.reduce_mean(M), 1e-1, 1e+1)
    sumLoss = tf.clip_by_value(sumLoss, 0.0, 1e+8)
    return sumLoss


def multiclassloss(M, G, R, weights):
    diff = G.get_shape().as_list()[1]
    labels = tf.argmax(R, axis=1)
    one_hot = tf.one_hot(labels, depth=diff, dtype=tf.float32)
    class_weights = tf.constant(weights)
    weights = tf.cast(tf.gather(class_weights, labels), tf.float32)
    unweighted_losses = (
        tf.nn.softmax_cross_entropy_with_logits(
            labels=M * one_hot, logits=M * G))
    SparseCE = unweighted_losses * weights
    sumLoss = tf.reduce_mean(SparseCE)
    sumLoss /= tf.clip_by_value(tf.reduce_mean(M), 1e-1, 1e+1)
    sumLoss = tf.clip_by_value(sumLoss, 0.0, 1e+8)
    return sumLoss


def mseloss(M, G, R):
    sumLoss = tf.reduce_mean(M * tf.square(R - G))
    sumLoss /= tf.clip_by_value(tf.reduce_mean(R), 1e-1, 1e+8)
    sumLoss = tf.clip_by_value(sumLoss, 0.0, 1e+8)
    return sumLoss


def calculate_rmse_pfc(not_missing_data: table,
                       imputed_data: table,
                       missing_matrix: np.ndarray,
                       in_var: list, num_var: list, fac_var: list) -> list:
    pfcs = []
    rmses = []
    for idx, col in enumerate(in_var):
        if col in num_var:
            only_imputed = imputed_data.loc[missing_matrix[:, idx], col].values
            only_real = not_missing_data.loc[missing_matrix[:, idx], col].values
            try:
                result = np.mean(np.square(only_real - only_imputed))
                a = np.sqrt(np.mean(np.square(only_real - only_imputed)))
            except Exception as e:
                a = 0
                pass
            if np.isnan(a):
                a = 0
            else:
                pass
            rmses.append(a)
        else:
            only_imputed = imputed_data.loc[missing_matrix[:, idx], col].values
            only_real = not_missing_data.loc[missing_matrix[:, idx], col].values
            a = np.sum(only_real != only_imputed) / len(only_real)
            if np.isnan(a):
                a = 0
            else:
                pass

            pfcs.append(a)
    #             print(f"PFC Loss : {col}, {a}")
    return sum(pfcs), sum(rmses)


class Com:
    def W_Init(self, act):
        if act in [tf.nn.leaky_relu, tf.nn.relu]:
            init = self.relu_w_init
        elif act in [tf.nn.tanh, tf.nn.softmax, tf.nn.sigmoid]:
            init = self.tanh_w_init
        elif act in [tf.nn.selu, tf.nn.elu]:
            init = self.s_elu_w_init
        else:
            init = self.s_elu_w_init
        return init

    def printProgress(self, iteration, total, prefix='', suffix='', decimals=1, barLength=50):
        formatStr = "{0:." + str(decimals) + "f}"
        percent = formatStr.format(100 * (iteration / float(total)))
        filledLength = int(round(barLength * iteration / float(total)))
        bar = '#' * filledLength + '-' * (barLength - filledLength)
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def sample_Z(self, m, n):
        return np.random.uniform(0., 1, size=[m, n])

    def sample_M(self, m, n, p):
        A = np.random.uniform(0., 1., size=[m, n])
        B = A > p
        C = 1. * B
        return C

    def sample_idx(self, m, n):
        A = np.random.permutation(m)
        idx = A[:n]
        return idx

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def max_norm_regularizer(self, threshold, axes=1, name="max_norm", collection="max_norm"):
        def max_norm(weights):
            clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
            clip_weights = tf.assign(weights, clipped, name=name)
            tf.add_to_collection(collection, clip_weights)
            return None  # there is no regularization loss term

        return max_norm

    def spectral_norm(self, w, iteration=2, name=None):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        u = tf.get_variable(name, [1, w_shape[-1]],
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

    def CatNumEmb(self, out, cond, key_cond):
        rawinputs = []
        onehotinputs = []
        arginputs = []
        keys = list(key_cond.keys())
        if len(cond) == cond[-1][1]:
            self.log("All Numeric")
            Input = self.G_final_Act(out)
            Arg = self.G_final_Act(out)
        else:
            for idx, c in enumerate(cond):
                StartNode, TerminalNode = c[0], c[1]
                diff = TerminalNode - StartNode
                split = tf.slice(out, [0, StartNode], [-1, diff])  #
                if diff == 1:
                    type_ = "numeric"
                    __float__ = self.G_final_Act(split)
                    rawinputs.append(__float__)
                    onehotinputs.append(__float__)
                    arginputs.append(__float__)
                else:
                    prob = tf.nn.softmax(split)
                    arg = tf.argmax(prob, axis=1)
                    onehot = tf.one_hot(arg, depth=diff,
                                        dtype=tf.float32)
                    arg = tf.expand_dims(tf.cast(arg, tf.float32), axis=1)
                    rawinputs.append(prob)
                    onehotinputs.append(onehot)
                    arginputs.append(arg)
                    if diff == 2:
                        type_ = "binary"
                    else:
                        type_ = "multiclass"
                print("Column : {:20} | {:10} | [{}]".format(keys[idx], type_, diff))
            Input = tf.concat(rawinputs, axis=1, name='Inputs')
            onehotInput = tf.concat(onehotinputs, axis=1, name='oneHotInputs')
            argInput = tf.concat(arginputs, axis=1, name='Args')
        return Input, onehotInput, argInput

    def CatNumEmb_Loss(self, Gene, Real, Mask, cond, key_cond, Weight_Info):
        LOSS = 0
        keys = list(key_cond.keys())
        self.seperate_var = []
        with tf.variable_scope("Columns/Loss"):
            if len(cond) == cond[-1][1]:
                LOSS = gainloss(Mask, Gene, Real, None)
                for idx, (c, w_info) in enumerate(zip(cond, Weight_Info)):
                    StartNode, TerminalNode = c[0], c[1]
                    diff = TerminalNode - StartNode
                    Gsplit = tf.slice(Gene, [0, StartNode], [-1, diff])
                    Rsplit = tf.slice(Real, [0, StartNode], [-1, diff])
                    Msplit = tf.slice(Mask, [0, StartNode], [-1, diff])
                    type_ = "mse"
                    sumLoss = gainloss(Msplit, Gsplit, Rsplit, None)
                    print("StartNode : {:3} | {:10} | [{}]". \
                          format(StartNode, type_, diff))
                    # setattr(mod, 'loss_{}'.format(keys[idx]), sumLoss)
                    self.seperate_var.append(sumLoss)
            else:
                for idx, (c, w_info) in enumerate(zip(cond, Weight_Info)):
                    StartNode, TerminalNode = c[0], c[1]
                    diff = TerminalNode - StartNode
                    Gsplit = tf.slice(Gene, [0, StartNode], [-1, diff])
                    Rsplit = tf.slice(Real, [0, StartNode], [-1, diff])
                    Msplit = tf.slice(Mask, [0, StartNode], [-1, diff])
                    sumLoss = gainloss(Msplit, Gsplit, Rsplit, w_info)
                    if diff == 1:
                        type_ = "mse"
                    elif diff == 2:
                        type_ = "binary"
                    else:
                        type_ = "multiclass"
                    LOSS += sumLoss
                    print("StartNode : {:3} | {:10} | [{}]".format(StartNode, type_, diff))
                    # import sys
                    # mod = sys.modules[__name__]
                    # setattr(mod, 'loss_{}'.format(keys[idx]), sumLoss)
                    self.seperate_var.append(sumLoss)
        return LOSS

    def calculate_rmse_pfc(self,
                           not_missing_data: table,
                           imputed_data: table,
                           missing_matrix: np.ndarray,
                           in_var: list, num_var: list, fac_var: list) -> list:
        results = []
        for idx, col in enumerate(in_var):
            if col in num_var:
                only_imputed = imputed_data.loc[missing_matrix[:, idx], col].values
                only_real = not_missing_data.loc[missing_matrix[:, idx], col].values
                try:
                    result = np.mean(np.square(only_real - only_imputed))
                    a = np.sqrt(np.mean(np.square(only_real - only_imputed)))
                except Exception as e:
                    a = 0
                    pass
            else:
                only_imputed = imputed_data.loc[missing_matrix[:, idx], col].values
                only_real = not_missing_data.loc[missing_matrix[:, idx], col].values
                a = np.sum(only_real != only_imputed) / len(only_real)
            #             print(f"PFC Loss : {col}, {a}")
            if np.isnan(a):
                a = 0
            else:
                pass
            results.append(a)
        pfcs = results[len(num_var):]
        rmses = results[:len(num_var)]
        return sum(pfcs), sum(rmses)

#     def CatNumEmb_Loss(self ,  Gene , Real, M , cond , key_cond , seg ) :
#         condition = cond
#         origin_col = self.Original_Column
#         LOSS = 0
#         cond = condition + [ self.total]
#         with tf.variable_scope("Columns/Loss"):
#             if len(condition) == self.total :
#                 sumLoss = tf.reduce_mean(M * tf.square( Real - Gene ))
#                 sumLoss /= tf.clip_by_value(tf.reduce_mean(M),1e-8,100)
#                 LOSS = tf.clip_by_value(sumLoss,0.0,100)
#                 if seg == "Train" :
#                     print("======================")
#                     print(" Only Numeric Columns ")
#                     print("======================")
#                 tf.summary.scalar("TOTAL_MSE", LOSS )
#             else :
#                 for idx in range(len(condition)) :
#     #                 try :
#                     diff = cond[idx+1] - cond[idx]
#                     Gsplit = tf.slice(Gene , [0 , cond[idx] ] ,
#                                      [self.batch_size , diff] ) #
#                     Rsplit = tf.slice(Real , [0 , cond[idx] ] ,
#                                      [self.batch_size , diff] ) #
#                     Msplit = tf.slice(M , [0 , cond[idx] ] ,
#                                      [self.batch_size , diff] ) #
#                     if diff == 1 :
#                         type_ = "mse"
#                         #Missing = 1-Msplit
#                         Missing = Msplit
#                         fake = self.G_final_Act(Gsplit)
#                         real = Rsplit
#                         sumLoss = tf.reduce_mean(Missing * tf.square( real - fake ))
#                         sumLoss /= tf.clip_by_value(tf.reduce_mean(Missing),1e-8,100)
#                         sumLoss = tf.clip_by_value(sumLoss,0.0,100)

#                     elif diff == 2 :
#                         type_ = "binary"
#                         #Missing = 1-Msplit
#                         Missing = Msplit
#                         one_hot = tf.one_hot(tf.argmax( Rsplit , 1 ), depth = diff )
#                         """
#                         https://alexisalulema.com/2017/12/15/classification-loss-functions-part-ii/
#                         """
#                         weights = self.missing_info["obj_weight_info"][idx]
#                         weight = tf.constant( [weights[0]/weights[1]] )
#                         WCE = tf.nn.weighted_cross_entropy_with_logits(targets = one_hot ,
#                                                                        logits = Gsplit ,pos_weight =  weight)
#                         #CE = tf.nn.sigmoid_cross_entropy_with_logits(logits = Gsplit , labels = one_hot )
#                         sumLoss =tf.reduce_mean(Missing * WCE )
#                         sumLoss /= tf.clip_by_value(tf.reduce_mean(Missing), 1e-8 , 1000)
#                         sumLoss = tf.clip_by_value(sumLoss,0.0,100)
#                     else :
#                         type_ = "multiclass"
#                         #Missing = 1-Msplit
#                         Missing = Msplit
#                         one_hot = tf.one_hot(tf.argmax(Rsplit , 1), depth = diff )
#                         Missing = tf.reduce_mean(Missing,axis = 1 )
#                         ## softmax
# #                         sumLoss =\
# #                         tf.reduce_mean(Missing * tf.nn.softmax_cross_entropy_with_logits(
# #                             logits = Gsplit , labels = one_hot))
#                         ## https://code-examples.net/ko/q/2a7f0a5
#                         ## sparse softmax
#                         weights = self.missing_info["obj_weight_info"][idx]
#                         class_weights = tf.constant(weights)
#                         labels = tf.argmax(Rsplit, axis = 1 )
#                         weights = tf.gather(class_weights,  labels)
#                         SparseCE = tf.losses.sparse_softmax_cross_entropy(labels, Gsplit , weights)
#                         sumLoss =tf.reduce_mean(Missing * SparseCE)
#                         ## focal softmax
# #                         labels = tf.argmax(Rsplit, axis = 1 )
# #                         FocalSoftmax =focal_loss_softmax(labels=labels , logits=Gsplit)
# #                         sumLoss =tf.reduce_sum(Missing * FocalSoftmax)

#                         sumLoss /= tf.clip_by_value(tf.reduce_mean(Missing),1e-8,100)
#                         sumLoss = tf.clip_by_value(sumLoss,0.0,100)
#                     colLoss = sumLoss
#                     if seg == "Train" :
#                         print("Columns : {:25} | {:10} | [{}]".format(origin_col[idx] , type_ , diff))
#                     tf.summary.scalar(str(diff)+"_"+origin_col[idx].replace(" ","_") , colLoss )
#                     LOSS += colLoss
#         return LOSS