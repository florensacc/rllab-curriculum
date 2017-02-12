'''code originally from khaotik at https://github.com/khaotik/char-rnn-tensorflow

to be modified by leavesbreathe'''
import tensorflow as tf
from math import pi

def abs2_c(z):
    return tf.real(z)*tf.real(z)+tf.imag(z)*tf.imag(z)

def complex_mul_real( z, r ):
    return tf.complex(tf.real(z)*r, tf.imag(z)*r)


def refl_c(in_, normal_):
    normal_rk2 = tf.expand_dims( normal_, 1 )
    scale = 2*tf.matmul( in_, tf.conj( normal_rk2 ) )
    return in_ - tf.matmul(scale, tf.transpose(normal_rk2))


#get complex variable
def get_variable_c( name, shape, initializer=None ):
    re = tf.get_variable(name+'_re', shape=shape, initializer=initializer)
    im = tf.get_variable(name+'_im', shape=shape, initializer=initializer)
    return tf.complex(re,im, name=name)


#get unit complex numbers in polar form
def get_unit_variable_c( name, scope, shape ):
    theta = tf.get_variable(name, shape=shape, initializer = tf.random_uniform_initializer(-pi,pi) )
    return tf.complex( tf.cos(theta), tf.sin(theta) )


def modrelu_c(in_c, bias):
    if not in_c.dtype.is_complex:
        raise(ValueError('modrelu_c: Argument in_c must be complex type'))
    if bias.dtype.is_complex:
        raise(ValueError('modrelu_c: Argument bias must be real type'))
    n = tf.complex_abs(in_c)
    scale = 1./(n+1e-5)
    return complex_mul_real(in_c, ( tf.nn.relu(n+bias)*scale ))

def normalize_c(in_c):
    norm = tf.reduce_sum(
            abs2_c(in_c),
            reduction_indices=len(in_c.get_shape().as_list())-1
            )
    return tf.transpose(complex_mul_real(tf.transpose(in_c),1./(1e-5+tf.transpose(norm))))

