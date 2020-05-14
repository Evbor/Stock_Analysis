import os
import json
import tensorflow as tf
from stockanalysis.data import load_df
from stockanalysis.train import train
from stockanalysis.models import model_0
from stockanalysis.preprocess import preprocess

########################################
## Functions for running ML pipelines ##
########################################

def run_pipelines(path_to_data, model_name, model_version, gpu_memory, custom, **kwargs):
    """
    Configures and runs ML pipelines, verifying whether the dataset stored in
    the directory pointed to by :param path_to_data: works with the ML pipeline
    requested by the :param custom: flag.

    :param path_to_data: string, path to the directory where the dataset to run
                         the ML pipeline on is stored
    :param model_name: string, name of the ML model to be trained by the
                       request ML pipeline
    :param model_version: int >= 0, version number of the ML model to be trained
                          by the requested ML pipeline
    :param gpu_memory: int >=0 or None, amount of GPU memory to allocate for
                       training models
    :param custom: bool, specifies whether to use the custom ML pipeline
                   implemented in this module, or the default ML pipeline
                   implemented as the function pipeline in this module
    :**kwargs: the parameters for the ML pipeline to be run. All pipelines
               must implement :param path_to_data:, :param model_name:,
               :param model_version:, and :param gpu_memory: parameters. Other
               parameters are passed as **kwargs, to the pipeline ran.

    ---> None
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

############################################
## END Functions for running ML pipelines ##
############################################


########################################
## Default ML pipeline implementation ##
########################################

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
    Default ML pipeline implementation. Loads the dataset, preprocess the
    dataset, trains a ML model on the preprocessed data, and then saves the
    trained model to disk. For more information on the preprocessing, and training
    processes, along with the architecture of models trained see: (link)

    :param path_to_data: string, path to the directory that stores the data to
                         be consumed by this pipeline
    :param model_name: string, name to save the trained ML model as
    :param model_version: string, version number to save the trained ML model as
    :param model_version: int >= 0, version number of the ML model to be trained
                          by the requested ML pipeline
    :param tickers: string, stock tickers of the dataset to train the model on
    :param seed: int, random seed to set for the whole ML pipeline

    ---> None
    """

    df = load_df(path_to_data)

    # Preprocessing Data
    preprocess_params = {
                         'tickers': tickers,
                         'cut_off': 18,
                         'window_size': 5,
                         'seed': seed,
                         'norm_dirname': 'norm_full'
                         'encode_dirname': 'encode_full'
                         }

    (X, y), vocab = preprocess2(df, **preprocess_params)

    # Setting Model's Hyperparameters
    output_bias_init = {key: y[key].mean() for key in y}
    model_params = {
                    'output_bias_init': output_bias_init,
                    'vocab': vocab,
                    'doc_embedding_size': 100,
                    'lstm_layer_units': 32
                    }
    training_params = {
                       'batch_size': 6,
                       'epochs': 10,
                       'callbacks': []
                       }
    loss = tf.keras.losses.MeanSquaredError
    optimizer = tf.keras.optimizers.Adam
    optimizer_params = {}

    model_version = 'final'
    hyperparameters = {
                       'model_parameters': model_params,
                       'training_parameters': training_params,
                       'loss': loss,
                       'optimizer': optimizer,
                       'optimizer_parameters': optimizer_params,
                       'version': model_version
                       }
    metrics = []
    run_number = 0

    # Configuring Virtual GPU and hardware
    config_hardware(gpu_memory, seed)

    # Building and Training Model
    model = train(model_0, hyperparameters, metrics, run_number, X, y)

    model.save(os.path.join('models', model_name, str(model_version)))

############################################
## END Default ML pipeline implementation ##
############################################


#######################################
## Custom ML pipeline implementation ##
#######################################

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

###########################################
## END Custom ML pipeline implementation ##
###########################################
