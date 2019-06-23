#! -*- coding: utf-8 -*-
# refer: https://kexue.fm/archives/5112
# import from https://github.com/bojone/Capsule/blob/master/Capsule_Keras.py

from keras import activations
from keras import backend as K
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers import *

from keras.utils.generic_utils import get_custom_objects

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x


#define our own softmax function instead of K.softmax
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)


#A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule=7, dim_capsule=16, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, 1)
            o = K.batch_dot(c, u_hat_vecs, [2, 2])
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                b = K.batch_dot(o, u_hat_vecs, [2, 3])
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

class CapsuleNet(object):
    def __init__(self, size, class_num):
        self.size = size
        self.class_num = class_num

    def __call__(self):
        input_image = Input(shape=self.size)
        cnn = Conv2D(64, (5, 5), padding='same', activation='relu')(input_image)
        cnn = Conv2D(64, (5, 5), padding='same', activation='relu')(cnn)
        cnn = MaxPooling2D((2,2))(cnn)
        cnn = Conv2D(128, (5, 5), padding='same', activation='relu')(cnn)
        cnn = Conv2D(128, (5, 5), padding='same', activation='relu')(cnn)
        cnn = MaxPooling2D((2,2))(cnn)
        cnn = Conv2D(256, (5, 5), padding='same', activation='relu')(cnn)
        cnn = Conv2D(256, (5, 5), padding='same', activation='relu')(cnn)
        cnn = MaxPooling2D((2,2))(cnn)
        cnn = Conv2D(512, (5, 5), padding='same', activation='relu')(cnn)
        cnn = Conv2D(512, (5, 5), padding='same', activation='relu')(cnn)
        cnn = Reshape((-1, 512))(cnn)
        capsule = Capsule(7, 16, 3, True)(cnn)
        output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(self.class_num,))(capsule)
        
        model = Model(inputs=input_image, outputs=output)

        return model

class CapsuleResNet(object):
    def __init__(self, size, class_num):
        self.size = size
        self.class_num = class_num

    def ResidualBlock(self, filters, inp, kernel_size=(3, 3), padding='same'):
        x = inp
        x = Conv2D(filters, kernel_size=kernel_size, padding=padding)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters, kernel_size=kernel_size, padding=padding)(x)
        x = Add()([x, inp])
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def __call__(self):
        input_image = Input(shape=self.size)
        cnn = Conv2D(32, (7, 7), padding='same')(input_image)
        cnn = LeakyReLU(alpha=0.2)(cnn)

        cnn = AveragePooling2D((2,2))(cnn)
        cnn = LeakyReLU(alpha=0.2)(cnn)

        cnn = Conv2D(64, (5, 5), padding='same')(cnn)
        cnn = LeakyReLU(alpha=0.2)(cnn)
        cnn = self.ResidualBlock(64, cnn)
        cnn = self.ResidualBlock(64, cnn)

        cnn = AveragePooling2D((2,2))(cnn)
        cnn = LeakyReLU(alpha=0.2)(cnn)

        cnn = Conv2D(128, (5, 5), padding='same')(cnn)
        cnn = LeakyReLU(alpha=0.2)(cnn)
        cnn = self.ResidualBlock(128, cnn)
        cnn = self.ResidualBlock(128, cnn)

        cnn = AveragePooling2D((2,2))(cnn)
        cnn = LeakyReLU(alpha=0.2)(cnn)

        cnn = Conv2D(256, (5, 5), padding='same')(cnn)
        cnn = LeakyReLU(alpha=0.2)(cnn)
        cnn = self.ResidualBlock(256, cnn)
        cnn = self.ResidualBlock(256, cnn)

        cnn = Reshape((-1, 256))(cnn)
        capsule = Capsule(7, 16, 3, True)(cnn)
        output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(self.class_num,))(capsule)
        
        model = Model(inputs=input_image, outputs=output)

        return model

get_custom_objects().update({'Capsule': Capsule})
