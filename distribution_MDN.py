#imports
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline

import math
import random
import time
import os
import cPickle as pickle

import tensorflow as tf #built with TensorFlow version 1.0

#mdn networks
#put a dense cap on top of the rnn cells (to interface with the mixture density network)
n_out = 1 + args.nmixtures * 6 # params = end_of_stroke + 6 parameters per Gaussian
with tf.variable_scope('mdn_dense'):
    output_w = tf.get_variable("output_w", [args.rnn_size, n_out], initializer=model.graves_initializer)
    output_b = tf.get_variable("output_b", [n_out], initializer=model.graves_initializer)

output = tf.nn.xw_plus_b(out_cell2, output_w, output_b) #data flows through dense nn

#finishing
# ----- build mixture density cap on top of second recurrent cell
def gaussian2d(x1, x2, mu1, mu2, s1, s2, rho):
    # define gaussian mdn (eq 24, 25 from http://arxiv.org/abs/1308.0850)
    x_mu1 = tf.subtract(x1, mu1)
    x_mu2 = tf.subtract(x2, mu2)
    Z = tf.square(tf.div(x_mu1, s1)) + \
        tf.square(tf.div(x_mu2, s2)) - \
        2*tf.div(tf.multiply(rho, tf.multiply(x_mu1, x_mu2)), tf.multiply(s1, s2))
    rho_square_term = 1-tf.square(rho)
    power_e = tf.exp(tf.div(-Z,2*rho_square_term))
    regularize_term = 2*np.pi*tf.multiply(tf.multiply(s1, s2), tf.sqrt(rho_square_term))
    gaussian = tf.div(power_e, regularize_term)
    return gaussian
