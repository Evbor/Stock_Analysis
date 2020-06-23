import os
import json
import pickle
import numpy as np
import tensorflow as tf

#######################
## Model Definitions ##
#######################

# Dummy Model Definition

def baseline_model(output_bias):

    if output_bias is not None:
        output_bias_inits = {
                             tname: (tf.keras.initializers.Constant(bias)
                                     if bias is not None else None)
                             for tname, bias in output_bias.items()
                             }
    else:
        output_bias_inits = {'adjusted_close_WFC_target': None}

    inputs = {
              'adjusted_close_WFC': tf.keras.Input(shape=(5,),
                                                   name='adjusted_close_WFC',
                                                   dtype=tf.float32)
             }

    features = inputs['adjusted_close_WFC']

    outputs = {
               'adjusted_close_WFC_target': tf.keras.layers.Dense(1, name='adjusted_close_WFC_target', kernel_initializer='zeros', bias_initializer=output_bias_inits['adjusted_close_WFC_target'])(features)
               }

    model = tf.keras.Model(inputs, outputs, name='baseline_model')

    return model

def val_schema_baseline_model(schema):
    needed_features = {
                       'adjusted_close_WFC'
                       }
    if not needed_features.issubset(schema):
        raise RuntimeError('Data schema mismatch for the selected model. Model requires features with names: {}'.format(', '.join(needed_features)))

# END Dummy Model Definition

###########################
## END Model Definitions ##
###########################
