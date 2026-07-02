import tensorflow as tf

def res_block(x, filters, downsample= False):
    skip= x
    stride = 2 if downsample else 1
        
    if downsample or x.shape[-1] != filters:
        skip= tf.keras.layers.Conv2D(filters, (1,1), strides= stride, padding='valid', use_bias= False)(skip)
        skip= tf.keras.layers.BatchNormalization()(skip)
        
    x= tf.keras.layers.Conv2D(filters, (3,3), strides=stride, padding= 'same', use_bias= False)(x)
    x= tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)
    
    x= tf.keras.layers.Conv2D(filters, (3,3), strides=1, padding= 'same', use_bias= False)(x)
    x= tf.keras.layers.BatchNormalization()(x)     
    
    x= tf.keras.layers.Add()([x, skip])
    x= tf.keras.layers.ReLU()(x)

    return x