import os
import json
import pandas as pd
import tensorflow as tf
from stockanalysis.train import train
from stockanalysis.models import model_0
from stockanalysis.preprocess import preprocess

### TODO:
# 1) Figure out where the default pipeline is breaking/fix it.
# 2) Write better docstrings
# 3) determine the pros and cons of configuring the GPU outside of the pipeline
#    if the cons aren't bad implement configuring the GPU in where it is commented.


def run_pipelines(path_to_data, gpu_memory, custom, **kwargs):
    """
    Configures and runs ML pipelines.
    """
    
    if custom:
        if verify_custom(path_to_data, **kwargs):
            # Config GPU
            custom_pipeline(path_to_data, **kwargs)
        else:
            print('Current custom pipeline configuration cannot handle the data stored in: {}'.format(path_to_data))
    else:
        if verify(path_to_data, **kwargs):
            # Config GPU
            pipeline(path_to_data, gpu_memory, **kwargs)
        else:
            print('Current pipeline configuration cannot handle the data stored in: {}'.format(path_to_data))

def verify(path_to_data, **kwargs):
    """
    Verifies if the data stored in the location :param path_to_data: can be
    consumed by the default pipeline.

    :param path_to_data: string, path to the data file
    :params **kwargs: configurable parameters used to configure the default
                      pipeline.

    ---> bool, True if the data can be used, False otherwise
    """

    legal_tickers = {'WFC', 'JPM', 'BAC', 'C'}
    required_text_data = '8-k'

    with open(os.path.join(path_to_data, 'meta.json')) as f:
        meta_data = json.load(f)

    return all((t in legal_tickers) and (t in meta_data['tickers']) for t in kwargs['tickers']) and (required_text_data in meta_data['form_types'])

def pipeline(path_to_data, gpu_memory, tickers, seed=None):
    """
    Docstring
    """

    # Accessing data storage medium
    df = pd.read_csv(os.path.join(path_to_data, 'raw.csv'),
                     parse_dates=['timestamp'])
    # Preprocessing Data
    preprocess_params = {'log_adj_ret_name': 'log_adj_daily_returns',
                         'feature_names': ['log_adj_daily_returns', 'docs'],
                         'label_feature_names': ['log_adj_daily_returns'],
                         'cut_off': 18, 'window_size': 5,
                         'norm_docs_fname': 'norm_test',
                         'path_to_vocab': os.path.join(path_to_data,
                                                       'vocab_test.json'),
                         'encode_docs_fname': 'encode_test'}

    (X, y), vocab = preprocess(df, preprocess_params, tickers, seed=seed)

    # Building model
    LOSS = tf.keras.losses.MeanSquaredError()
    OPTIMIZER = tf.keras.optimizers.Adam()
    model_params = {'lstm_layer_units': 256, 'vocab': vocab,
                    'doc_embedding_size': 200, 'output_bias_init': 0}
    training_params = {'batch_size': 4, 'epochs': 2}
    model_version = 'final'
    hyperparameters = {'model_parameters': model_params,
                       'training_parameters': training_params,
                       'loss': LOSS, 'optimizer': OPTIMIZER,
                       'version': model_version}
    metrics = []
    run_number = 1

    model = train(model_0, hyperparameters, metrics,
                  run_number, X, y, gpu_memory=gpu_memory, seed=seed)

    model.save(os.path.join('models', 'model_0', '2'))

def verify_custom(path_to_data, **kwargs):
    """
    Custom pipeline data verification function to implement.
    """
    return True

def custom_pipeline(path_to_data, **kwargs):
    """
    Custom pipeline to implement.
    """
    print('Custom pipeline is not implemented')
