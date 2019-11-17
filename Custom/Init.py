import tensorflow as tf

print(tf.__version__)
import tensorflow.contrib.slim as slim
import numpy as np
from scipy import arange

import sys
sys.path.append('/home/advice/Python/SR/Custom/')
from Activations import *
from moving_free_batch_normalization import moving_free_batch_norm
from stochastic_weight_averaging import StochasticWeightAveraging


def focal_loss_sigmoid(labels,logits,alpha=0.25 , gamma=2):
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-labels*(1-alpha)*((1-y_pred)*gamma)*tf.log(  tf.maximum(y_pred , 1e-14 )   )-\
      (1-labels)*alpha*(y_pred**gamma)*tf.log( tf.maximum( 1-y_pred ,  1e-14 ) ) 
    return L


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev)


def spectral_norm(w, iteration=1 , name = None):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable(name , [1, w_shape[-1]], 
                        initializer=tf.random_normal_initializer(), trainable=False)
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

"""https://github.com/llealgt/NALUs"""

# class NALU:
# 	def __init__(self,input_shape=(0,0),size=2,epsilon = 1e-8,name = ""):
# 		print("hello NALU world "+name)
# 		self.size = size # the number of neurons or units in the NALU
# 		self.input_shape = input_shape # tuple describing the shape of the input to the NALU (observations,columns)
# 		self.epsilon = epsilon #used to avoid log of 0 
# 		self.name = name

# 		weights_shape = (self.input_shape[1],size)

# 		with tf.name_scope(name):
# 			self.W_hat = tf.get_variable(name+"W_hat",shape = weights_shape)
# 			self.M_hat = tf.get_variable(name+"M_hat",shape = weights_shape)
# 			self.G = tf.get_variable(name+"G",shape = weights_shape)
            
# 	def NALU_output(self,X):
# 		# NAC: a = Wx W = tanh(Wˆ ) * σ(Mˆ )

		
# 		W = tf.nn.tanh(self.W_hat) *  tf.nn.sigmoid(self.M_hat)
# 		a = tf.matmul(X,W)

# 		# NALU: y = g * a + (1 − g) *m  m = expW(log(|x| + epsilon)), g = σ(Gx)
# 		g = tf.nn.sigmoid(tf.matmul(X,self.G))
# 		m = tf.exp(tf.matmul(tf.log(tf.abs(X) + self.epsilon),W))

# 		y = (g*a) + (1-g)*m
# 		return y

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


def tf_nalu(input_layer, num_outputs , epsilon=1e-30 ):
    """ Neural Arithmetic Logic Unit tesnorflow layer
    Arguments:
    input_layer - A Tensor representing previous layer
    num_outputs - number of ouput units 
    Returns:
    A tensor representing the output of NALU
    https://github.com/grananqvist/NALU-tf/blob/master/nalu.py
    가장 쉽게 되는 듯??
    """
    shape = (int(input_layer.shape[-1]), num_outputs)

    # define variables
    W_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    M_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    G = tf.Variable(tf.truncated_normal(shape, stddev=0.02))

    # operations according to paper
    W = tf.tanh(W_hat) * tf.sigmoid(M_hat)
    m = tf.exp(tf.matmul(tf.log(tf.abs(input_layer) + epsilon ), W))
    g = tf.sigmoid(tf.matmul(input_layer, G))
    a = tf.matmul(input_layer, W)
    out = g * a + (1 - g) * m

    return out


def nac(x, num_outputs , name=None, reuse=None):
    """
    NAC as in https://arxiv.org/abs/1808.00508.    
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_layers.py
    """
    with tf.variable_scope(name, default_name="nac", values=[x], reuse=reuse):
        x_shape = shape_list(x)
        w = tf.get_variable("w", [x_shape[-1], num_outputs])
        m = tf.get_variable("m", [x_shape[-1], num_outputs])
        w = tf.tanh(w) * tf.nn.sigmoid(m)
        x_flat = tf.reshape(x, [-1, x_shape[-1]])
        res_flat = tf.matmul(x_flat, w)
        return tf.reshape(res_flat, x_shape[:-1] + [num_outputs])

def nalu(x, num_outputs, epsilon=1e-30, name=None, reuse=None):
    """
    NALU as in https://arxiv.org/abs/1808.00508.
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_layers.py
    """
    with tf.variable_scope(name, default_name="nalu", values=[x], reuse=reuse):
        x_shape = shape_list(x)
        x_flat = tf.reshape(x, [-1, x_shape[-1]])
        gw = tf.get_variable("w", [x_shape[-1], num_outputs])
        g = tf.nn.sigmoid(tf.matmul(x_flat, gw))
        g = tf.reshape(g, x_shape[:-1] + [num_outputs])
        a = nac(x, num_outputs, name="nac_lin")
        log_x = tf.log(tf.abs(x) + epsilon)
        m = nac(log_x, num_outputs, name="nac_log")
        return g * a + (1 - g) * tf.exp(m)

def sample_Z(m, n , type = "uniform"):
    if type == "uniform" :
        output = np.random.uniform(0., 1., size = [m, n]) 
    elif type == "normal" :
        output = np.random.normal(loc = 0., scale = 1., size = [m, n]) 
    return output

def log(x):
    return tf.log( tf.maximum( x , 1e-10) )


def linear(input, output_dim, scope=None, stddev=1.0):
    ## https://github.com/AYLIEN/gan-intro/blob/master/gan.py
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b

def layer(prev = None , shape1 = None , shape2 = None , 
          name = None , activation = tf.nn.leaky_relu , usebias = True , 
          final = False , SN = True , Type = None , phase = None ) :
    with tf.variable_scope(name):
        select_w_init = np.random.randint(0, 2, size=1 )[0]
        seed_n = np.random.randint(1, 1000, size=1 )[0]
        relu_w_init = [tf.keras.initializers.he_uniform(seed = seed_n) ,
                       tf.keras.initializers.he_normal(seed = seed_n)][select_w_init]
        tanh_w_init = [tf.keras.initializers.glorot_normal(seed = seed_n) ,
                       tf.keras.initializers.glorot_uniform(seed = seed_n)][select_w_init]
        s_elu_w_init = [tf.keras.initializers.lecun_normal(seed = seed_n) ,
                       tf.keras.initializers.lecun_uniform(seed = seed_n)][select_w_init]
        nomal_w_init = tf.keras.initializers.truncated_normal(seed = seed_n)
        if activation in [tf.nn.leaky_relu, tf.nn.relu] :  init = relu_w_init
        elif activation in [tf.nn.tanh , tf.nn.softmax] :  init = tanh_w_init
        elif activation in [tf.nn.selu , tf.nn.elu , mish] :      init = s_elu_w_init
        else : 
            if final : init = tanh_w_init
            else : init = nomal_w_init
        ###
        if usebias :b1 = tf.get_variable("Bias_" + str(name) , 
                                         shape = [shape2] , dtype = tf.float32 , 
                                         initializer = tf.constant_initializer(0.0))
        else :b1 = tf.constatnt(0.0 , shape = [shape2] , dtype = tf.float32 )

        W1 = tf.get_variable("Weight_" + str(name) , dtype = tf.float32 , 
                             shape = [shape1 , shape2] , initializer = init)
        
        
        if final == True :
            if SN == True :
                W2 = spectral_norm(W1 , name = "SN" + str(name))
                layer = tf.matmul( prev , W2) + b1
            else :
                layer = tf.matmul( prev , W1 ) + b1
        else :
            if SN == True :
                W2 = spectral_norm(W1 , name = "SN" + str(name))
                layer = tf.matmul( prev , W2) + b1
            else : layer = tf.matmul( prev , W1) + b1
            ################################
            if Type == "SWA" :
                layer = moving_free_batch_norm(layer, axis=-1, 
                                               training=is_training_bn,
                                               use_moving_statistics=use_moving_statistics, 
                                               momentum=0.99)
            elif Type == "Self_Normal" :
                if activation == nalu :layer = activation(layer ,2 , name = "NALU_" + name )
                else :layer = activation(layer)
                layer = tf.contrib.nn.alpha_dropout(layer , 0.8)
            elif Type == "Batch_Normalization" :
                layer = tf.contrib.layers.batch_norm(layer, 
                                                     center=True, scale=True, 
                                                     is_training=True, # phase
                                                     scope='bn')
            elif Type == "Instance_Normalization" :layer = tf.contrib.layers.instance_norm(layer)
            else : pass
            if Type == "Self_Normal" : pass
            else : 
                if activation == nalu :
                    layer = activation(layer ,2 , name = "NALU_" + name )
                else : layer = activation(layer)
        return layer

NUM_KERNELS = 5
def minibatch(input, num_kernels=NUM_KERNELS, kernel_dim=3, name = None , bs = None ):
    """https://github.com/AYLIEN/gan-intro/blob/master/gan.py"""
    output_dim = num_kernels*kernel_dim
    w = tf.get_variable("Weight_minibatch_" + name ,
                        [input.get_shape()[1], output_dim ],
                        initializer=tf.random_normal_initializer(stddev=0.2),
                        regularizer= slim.l2_regularizer(0.05) )
# tf.keras.regularizers.l2(0.05) # 
    b = tf.get_variable("Bias_minibatch_" + name ,
                        [output_dim],initializer=tf.constant_initializer(0.0))
    x = tf.matmul(input, w) + b
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    eps = tf.expand_dims(np.eye(int( bs ), dtype=np.float32), 1)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2) + eps
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    output = tf.concat([input, minibatch_features],1)
    return output

def buildMinibatchDiscriminator(features, numFeatures, kernels, kernelDim=5, reuse=False):
    """다른 버전 https://github.com/matteson/tensorflow-minibatch-discriminator/blob/master/discriminator.py"""
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        # TODO: no undefined dimensions until 1.0 release
        batchTensor = tf.get_variable('disc_minibatch',
                       shape=[numFeatures, kernels, kernelDim],
                       initializer=tf.truncated_normal_initializer(stddev=0.1),
                       regularizer= slim.l2_regularizer(0.05))
# tf.keras.regularizers.l2(0.05) # 

        #flatFeatures = slim.flatten(features)
        flatFeatures = tf.layers.flatten(features)
        multFeatures = tf.einsum('ij,jkl->ikl',flatFeatures, batchTensor)
        multFeaturesExpanded1 = tf.expand_dims(multFeatures,[1])

        fn = lambda x: x - multFeatures

        multFeaturesDiff = tf.exp(
            -tf.reduce_sum(
                tf.abs(
                    tf.map_fn(fn, multFeaturesExpanded1)
                ),
            axis=[3])
        )

        output = tf.reduce_sum(multFeaturesDiff, axis=[1]) - 1 # differs from paper, but convergence seems better with -1 in my experiments
    return output

def weights_init(shape):
    '''
    Weights initialization helper function.
    
    Input(s): shape - Type: int list, Example: [5, 5, 32, 32], This parameter is used to define dimensions of weights tensor
    
    Output: tensor of weights in shape defined with the input to this function
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def bias_init(shape, bias_init=0.05):
    '''
    Bias initialization helper function.
    
    Input(s): shape - Type: int list, Example: [32], This parameter is used to define dimensions of bias tensor.
              bias_value - Type: float number, Example: 0.01, This parameter is set to be value of bias tensor.
    
    Output: tensor of biases in shape defined with the input to this function
    '''
    return tf.Variable(tf.constant(bias_init, shape=shape))

def highway_fc_layer(input, shape1 = None , shape2 = None , carry_b = -2.0, activation=tf.nn.relu):
    '''
    The function used to crate Highway fully connected layer in the network.
    
    Inputs: input - data input
            hidden_layer_size - number of neurons in the hidden layers (highway layers)
            carry_b -  value for the carry bias used in transform gate
            activation - non-linear function used at this layer
    '''
    #Step 1. Define weights and biases for the activation gate
    weights_normal = weights_init([shape1, shape2])
    bias_normal = bias_init([shape2])
    #Step 2. Define weights and biases for the transform gate
    weights_transform = weights_init([shape1, shape2])
    bias_transform = bias_init(shape=[shape2], bias_init=carry_b)
    ## extra
    input_transform = weights_init([shape1, shape2])
    #Step 3. calculate activation gate
    H = activation(tf.matmul(input, weights_normal) + bias_normal, name="Input_gate")
    #Step 4. calculate transform game
    T = tf.nn.sigmoid(tf.matmul(input, weights_transform) +bias_transform, name="T_gate")
    #Step 5. calculate carry get (1 - T)
    C = tf.subtract(1.0, T, name='C_gate')
    # y = (H * T) + (x * C)
    #Final step 6. campute the output from the highway fully connected layer
    y = tf.add(tf.multiply(H, T), tf.multiply(tf.matmul(input,input_transform) , C), name='output_highway')
    return y
