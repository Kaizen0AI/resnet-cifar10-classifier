import tensorflow as tf
from src.blocks import res_block

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

def initial_resnet():
    input_x = tf.keras.layers.Input(shape= (32,32,3))
    x = data_augmentation(input_x)
    x = tf.keras.layers.Conv2D(24, (3,3), strides= 1, padding='same', use_bias= False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2))(x) # experiment 1; initial maxpooling

    x = res_block(x, 48, downsample= False)
    x = res_block(x, 96, downsample= True)
    x = res_block(x, 96, downsample= False)
    x = res_block(x, 160, downsample= True)
    x = res_block(x, 160, downsample= False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(10, activation= 'softmax')(x)
    model = tf.keras.Model(inputs= input_x, outputs= x)
    return model