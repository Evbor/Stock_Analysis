import os
import json
import pickle
import tensorflow as tf

#######################
## Model Definitions ##
#######################


# Dummy Model Definition

def baseline_model(output_bias_init=None):

    if output_bias_init is not None:
        output_bias = output_bias_init['adjusted_close_target_WFC']
        output_bias_init = tf.keras.initializers.Constant(output_bias)

    inputs = {
              'adjusted_close_WFC': tf.keras.Input(shape=(5,), name='adjusted_close_WFC', dtype=tf.float32),
              '8-k_WFC': tf.keras.Input(shape=(None,), name='8-k_WFC', dtype=tf.int64),
              'adjusted_close_JPM': tf.keras.Input(shape=(5,), name='adjusted_close_JPM', dtype=tf.float32),
              '8-k_JPM': tf.keras.Input(shape=(None,), name='8-k_JPM', dtype=tf.int64),
              'adjusted_close_BAC': tf.keras.Input(shape=(5,), name='adjusted_close_BAC', dtype=tf.float32),
              '8-k_BAC': tf.keras.Input(shape=(None,), name='8-k_BAC', dtype=tf.int64),
              'adjusted_close_C': tf.keras.Input(shape=(5,), name='adjusted_close_C', dtype=tf.float32),
              '8-k_C': tf.keras.Input(shape=(None,), name='8-k_C', dtype=tf.int64)
             }

    features = tf.keras.layers.Concatenate()([inputs[fname] for fname in inputs.keys() if '8-k' not in fname])

    output_layer = tf.keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer=output_bias_init,
                                         name='adjusted_close_target_WFC')

    outputs = {
               'adjusted_close_target_WFC': output_layer(features)
              }

    model = tf.keras.Model(inputs, outputs, name='baseline_model')

    return model

# END Dummy Model Definition

# Default Model Definitions

# END Default Model Definitions

###########################
## END Model Definitions ##
###########################
