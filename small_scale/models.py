import tensorflow as tf

def shared_model(input_shape, output_dim=200):
    
    inputs = tf.keras.layers.Input(shape=input_shape[1:], name='inputs')
    
    x = inputs
    x = tf.keras.layers.Conv2D(20, (5, 5))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(50, (5, 5))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(output_dim)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

def classification_model(inputs, num_classes, input_dim=200):
    inputs = tf.keras.layers.Input(shape=(input_dim), name='inputs')
    
    x = inputs
    
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)