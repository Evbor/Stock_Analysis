import os
import json
import pandas as pd
import tensorflow as tf
from stockanalysis.data import load_df
from stockanalysis.train import train
from stockanalysis.models import model_0
from stockanalysis.preprocess import preprocess

def run_pipelines(path_to_data, model_name, model_version, gpu_memory, custom, **kwargs):
    """
    Configures and runs ML pipelines.
    """

    if custom:
        if verify_custom(path_to_data, **kwargs):
            # Config GPU
            custom_pipeline(path_to_data, model_name, model_version, gpu_memory, **kwargs)
        else:
            print('Current custom pipeline configuration cannot handle the data stored in: {}'.format(path_to_data))
    else:
        if verify(path_to_data, **kwargs):
            # Config GPU
            pipeline(path_to_data, model_name, model_version, gpu_memory, **kwargs)
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


def pipeline(path_to_data, model_name, model_version, gpu_memory, tickers, seed):
    """
    Docstring
    """

    df = load_df(path_to_data)

    # Preprocessing Data
    preprocess_params = {'tickers': tickers, 'cut_off': 18,
                         'window_size': 5, 'seed': seed}
                         
    (X, y), vocab = preprocess(df, **preprocess_params)

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

    model.save(os.path.join('models', model_name, str(model_version)))

def verify_custom(path_to_data, **kwargs):
    """
    Custom pipeline data verification function to implement.
    """
    return True

def custom_pipeline(path_to_data, model_name, model_version, gpu_memory, **kwargs):
    """
    Custom pipeline to implement.
    """
    print('Custom pipeline is not implemented')
