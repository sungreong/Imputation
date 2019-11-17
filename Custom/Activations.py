import tensorflow as tf
import numpy as np
from scipy import arange

"""
Reference
https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_layers.py
"""
def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret

def brelu(x):
    """Bipolar ReLU as in https://arxiv.org/abs/1709.04054."""
    x_shape = shape_list(x)
    #x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 2]), 2, axis=-1)
    x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 1]), 2, axis=0)
    y1 = tf.nn.relu(x1)
    y2 = -tf.nn.relu(-x2)
    return tf.reshape(tf.concat([y1, y2], axis=-1), x_shape)


def belu(x):
    """Bipolar ELU as in https://arxiv.org/abs/1709.04054."""
    x_shape = shape_list(x)
    #x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 2]), 2, axis=-1)
    x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 1]), 2, axis=0)
    y1 = tf.nn.elu(x1)
    y2 = -tf.nn.elu(-x2)
    return tf.reshape(tf.concat([y1, y2], axis=-1), x_shape)


def bent_identity(x) :
    return tf.div(tf.sqrt(tf.square(x) + 1) -1 ,2) + x
    
    
def tf_mish(x) :
    return x * tf.nn.tanh(tf.nn.softplus(x))

def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
    x: float Tensor to perform activation.
    Returns:
    x with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.nn.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def mish(x) :
    return x * tf.nn.tanh( tf.nn.softplus(x))

def gauss(x) :
    """ 0 ~ 1 로 하는 가우시안 함수 """
    return tf.exp(-x**2)

def soft_cliping(alpha = 0.5 , x=None ) :
    """ 0 ~ 1 로 하는 새로운 함수 alpha라는 하이퍼 파라미터가 존재함"""
    first = tf.div(1.0 , alpha)
    second = tf.log( tf.div(tf.add(1.0, tf.exp( tf.multiply(alpha , x) )) , 
                            tf.add(1.0 , tf.exp( tf.multiply(alpha, (tf.add(x , -1.0)) )))))
    return tf.multiply(first , second )

## https://cup-of-char.com/writing-activation-functions-from-mostly-scratch-in-python/

def tf_sqnlsig(x):   #tensorflow SQNLsigmoid
    """https://cup-of-char.com/writing-activation-functions-from-mostly-scratch-in-python/"""
    u=tf.clip_by_value(x,-2,2)
    a = u
    b= tf.negative(tf.abs(u))
    wsq = (tf.multiply(a,b))/4.0
    y = tf.add(tf.multiply(tf.add(u,wsq),0.5),0.5)
    return y

#     alphas = tf.get_variable('alpha', _x.get_shape()[-1],
#                        initializer=tf.constant_initializer(0.0),
#                         dtype=tf.float32)
def parametric_relu(_x):
    alphas = tf.Variable(tf.zeros(_x.get_shape()[-1]),
                         name = "prelu" ,
                         dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


def tf_sqnl(x): #tensorflow SQNL
    """https://cup-of-char.com/writing-activation-functions-from-mostly-scratch-in-python/"""
    #tf.cond(x>2,lambda: tf.multiply(2,1),lambda:tf.multiply(x,1))
    #tf.cond(tf.less(x,-2),lambda: -2,lambda:tf.multiply(x,1))
    u=tf.clip_by_value(x,-2,2)
    a = u
    b= tf.negative(tf.abs(u))
    wsq = (tf.multiply(a,b))/4.0
    y = tf.add(u,wsq)
    return y

# 출처: https://creamyforest.tistory.com/48 [Dohyun's Blog]
def Relu2(x):
    return tf.minimum(tf.maximum(x ,0.0) , 2.0)