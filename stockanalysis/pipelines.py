import os
import pandas as pd
import tensorflow as tf
from stockanalysis.train import train
from stockanalysis.models import model_0
from stockanalysis.preprocess import preprocess


def pipeline(path_to_data, tickers, df_filename='raw.csv',
             gpu_memory=None, seed=None):
    """
    ML pipeline that trains a model, and saves it.
    """

    # Accessing data storage medium
    df = pd.read_csv(os.path.join(path_to_data, df_filename),
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
    return None
