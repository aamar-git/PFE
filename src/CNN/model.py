#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    19-Apr-2024 17:19:57

import tensorflow as tf
import keras
from keras import layers

def create_model():
    imageinput_unnormalized = keras.Input(shape=(98,50,1), name="imageinput_unnormalized")
    imageinput = keras.layers.Normalization(axis=(1,2,3), name="imageinput_")(imageinput_unnormalized)
    conv_1 = layers.Conv2D(12, (3,3), padding="same", name="conv_1_")(imageinput)
    batchnorm_1 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_1_")(conv_1)
    relu_1 = layers.ReLU()(batchnorm_1)
    maxpool_1 = layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")(relu_1)
    conv_2 = layers.Conv2D(24, (3,3), padding="same", name="conv_2_")(maxpool_1)
    batchnorm_2 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_2_")(conv_2)
    relu_2 = layers.ReLU()(batchnorm_2)
    maxpool_2 = layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")(relu_2)
    conv_3 = layers.Conv2D(48, (3,3), padding="same", name="conv_3_")(maxpool_2)
    batchnorm_3 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_3_")(conv_3)
    relu_3 = layers.ReLU()(batchnorm_3)
    maxpool_3 = layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")(relu_3)
    conv_4 = layers.Conv2D(48, (3,3), padding="same", name="conv_4_")(maxpool_3)
    batchnorm_4 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_4_")(conv_4)
    relu_4 = layers.ReLU()(batchnorm_4)
    conv_5 = layers.Conv2D(48, (3,3), padding="same", name="conv_5_")(relu_4)
    batchnorm_5 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_5_")(conv_5)
    relu_5 = layers.ReLU()(batchnorm_5)
    maxpool_4 = layers.MaxPool2D(pool_size=(13,1), strides=(1,1))(relu_5)
    dropout = layers.Dropout(0.200000)(maxpool_4)
    fc = layers.Reshape((-1,), name="fc_preFlatten1")(dropout)
    fc = layers.Dense(12, name="fc_")(fc)
    softmax = layers.Softmax()(fc)

    model = keras.Model(inputs=[imageinput_unnormalized], outputs=[softmax])
    return model
