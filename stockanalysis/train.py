import os
import re
import pickle
import tensorflow as tf

from copy import deepcopy

def config_hardware(gpu_memory, seed=None):
    """
    Configure GPU hardware if :param gpu_memory: is not None, else use CPU.

    :param gpu_memory: int or None, Number of bytes to allocate on the local
                       GPU for TensorFlow processes to use. If None, then use
                       CPU for TensorFlow processes.

    ---> None
    """

    if (gpu_memory == None) or (gpu_memory == 0):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    visible_gpus = tf.config.experimental.get_visible_devices('GPU')
    print('GPUs: {}'.format(gpus))
    print('Visible GPUs: {}'.format(visible_gpus))

    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    # Setting Tensorflow Seed
    tf.random.set_seed(seed)

    return None

def write_hparams(path_to_logs, model_name, hparams, verbose=True):
    version_number = hparams['version']
    path = os.path.join(path_to_logs, 'models', model_name,
                        '_'.join(['version', str(version_number)]))
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'hparams.pickle'), 'wb') as f:
        pickle.dump(hparams, f)
    if verbose:
        return print('Saved hyperparameters to file: {}'.format(path))
    else:
        return None

def build_compiled_model(build_model, hparams, metrics, run_number):
    """
    Builds compiled model from build model function.

    :param build_model: function, the function that returns a uncompiled model
    :param hparams: dict, key word indexed hyperparamters for constructing the
                    model defined by :param build_model:
    :param metrics: tensorflow.metrics, to compile the model with
    :param run_number: int, run number uniquely identifying the returned
                       compiled model, used loading a model from a saved
                       checkpoint

    ---> tensorflow.keras.Model, that is compiled and ready to fit on a dataset
         or predict
    """
    model_name = build_model.__name__
    hparam_version = hparams['version']
    loss = hparams['loss']()
    optimizer = hparams['optimizer'](**hparams['optimizer_parameters'])
    model_parameters = hparams['model_parameters']
    model = build_model(**model_parameters)

    ckpt_dir = os.path.join('logs', 'models', model_name,
                            '_'.join(['version', str(hparam_version)]),
                            'runs', str(run_number), 'checkpoints')
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if latest_ckpt:
        initial_epoch = re.findall(r'cp-(\d+)\.ckpt', latest_ckpt)[0]
        initial_epoch = int(initial_epoch) - 1
        model.load_weights(latest_ckpt)
        print('Restored model from: {}'.format(latest_ckpt))
    else:
        initial_epoch = 0

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model, initial_epoch

def train(build_model, hparams, metrics, run_number, X=None, y=None,
          validation_data=None, write_hyperparameters=False,
          checkpoint=False, log=False):
    """
    Builds and trains model defined by :build_model:

    :param build_model: function, the function that returns a uncompiled model
    :param hparams: dict, key word indexed hyperparamters for constructing the
                    model defined by :param build_model:
    :param metrics: tensorflow.metrics, to compile the model with
    :param run_number: int, run number uniquely identifying the returned
                       compiled model, used when loading a model from a
                       saved checkpoint
    :param X and y: dataset, where X are the features, and y are the labels to
                    train models on. See TensorFlow 2 Keras Model.fit
                    documentation for compatibable object types
    :param validation_data: dataset, to validated model on. See Tensorflow 2
                            Keras Model.fit documentation for compatible object
                            types
    :param write_hyperparameters: bool, set to True to write
                                  :param hyperparameters: to disk at
                                  /logs/models/[MODEL NAME]/
                                  version_[VERSION NUMBER]/:param run_number:/
                                  hparams.pickle. Where [MODEL NAME] is the
                                  name attribute of :param build_model: and
                                  [VERSION NUMBER] is the version attribute of
                                  the hyperparameters object.
    :param checkpoint: bool, set to True to checkpoint the model during training
                       at /logs/models/[MODEL NAME]/version_[VERSION NUMBER]/
                       :param run_number:/checkpoints/
    :param log: bool, set to True to log metrics and losses in a csv file
                during training at /logs/models/[MODEL NAME]/
                version_[VERSION NUMBER]/:param run_number:/history.log

    ---> tensorflow.keras.Model, tensorflow.keras.History, a tuple of the trained
         model, along with its training history
    """

    # Building Model
    model, initial_epoch = build_compiled_model(build_model, hparams, metrics, run_number)

    # Unpacking model training parameters
    ## Must deep copy the training parameters dictionary because otherwise var training_parameters is just pointer to
    ## the 'training_parameters' location of the hyperparameters object. This makes hyperparameters mutable,
    ## causing the hyperparameters to be locked during training because we add the mutable checkpointing callbacks
    ## this makes hyperparameters unpickleable on further reruns in in the same session.
    training_parameters = deepcopy(hparams['training_parameters'])

    assert initial_epoch != training_parameters['epochs'] - 1

    # Setting up checkpointing callbacks
    if checkpoint:
        path_to_run = os.path.join('logs', 'models', build_model.__name__, '_'.join(['version', str(hparams['version'])]), 'runs', str(run_number))
        path_to_ckpts = os.path.join(path_to_run, 'checkpoints')
        if not os.path.exists(path_to_ckpts):
            os.makedirs(path_to_ckpts)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(path_to_ckpts, 'cp-{epoch}.ckpt'), verbose=1, save_weights_only=True)
        if 'callbacks' not in training_parameters:
            training_parameters['callbacks'] = []
        training_parameters['callbacks'].append(cp_callback)
    if log:
        path_to_run = os.path.join('logs', 'models', build_model.__name__, '_'.join(['version', str(hparams['version'])]), 'runs', str(run_number))
        csv_logger = tf.keras.callbacks.CSVLogger(filename=os.path.join(path_to_run, 'history.log'), append=True)
        if 'callbacks' not in training_parameters:
            training_parameters['callbacks'] = []
        training_parameters['callbacks'].append(csv_logger)

    # Writing Hyperparameters
    if write_hyperparameters:
        # Saving Model's Hyperparameters
        write_hparams(build_model.__name__, hparams)

    # Training Model
    model_history = model.fit(X, y, **training_parameters, initial_epoch=initial_epoch, validation_data=validation_data)

    return model, model_history



def build_compiled_model2(build_model, hparams, metrics, run_number,
                          path_to_logs):
    """
    Builds compiled model from build model function.

    :param build_model: function, the function that returns a uncompiled model
    :param hparams: dict, key word indexed hyperparamters for constructing the
                    model defined by :param build_model:
    :param metrics: tensorflow.metrics, to compile the model with
    :param run_number: int, run number uniquely identifying the returned
                       compiled model, used loading a model from a saved
                       checkpoint

    ---> tensorflow.keras.Model, that is compiled and ready to fit on a dataset
         or predict
    """

    hparam_version = hparams['version']
    loss = hparams['loss']()
    optimizer = hparams['optimizer'](**hparams['optimizer_parameters'])
    model_parameters = hparams['model_parameters']
    model = build_model(**model_parameters)

    model_name = model.name

    ckpt_dir = os.path.join(path_to_logs, 'models', model_name,
                            '_'.join(['version', str(hparam_version)]),
                            'runs', str(run_number), 'checkpoints')
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if latest_ckpt:
        initial_epoch = re.findall(r'cp-(\d+)\.ckpt', latest_ckpt)[0]
        initial_epoch = int(initial_epoch) - 1
        model.load_weights(latest_ckpt)
        print('Restored model from: {}'.format(latest_ckpt))
    else:
        initial_epoch = 0

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model, initial_epoch


def train2(build_model, hparams, metrics, run_number, X=None, y=None,
           validation_data=None, path_to_logs=None):
    """
    Builds and trains model defined by :build_model:

    :param build_model: function, the function that returns a uncompiled model
    :param hparams: dict, key word indexed hyperparamters for constructing the
                    model defined by :param build_model:
    :param metrics: tensorflow.metrics, to compile the model with
    :param run_number: int, run number uniquely identifying the returned
                       compiled model, used when loading a model from a
                       saved checkpoint
    :param X and y: dataset, where X are the features, and y are the labels to
                    train models on. See TensorFlow 2 Keras Model.fit
                    documentation for compatibable object types
    :param validation_data: dataset, to validated model on. See Tensorflow 2
                            Keras Model.fit documentation for compatible object
                            types
    :param write_hyperparameters: bool, set to True to write
                                  :param hyperparameters: to disk at
                                  /logs/models/[MODEL NAME]/
                                  version_[VERSION NUMBER]/:param run_number:/
                                  hparams.pickle. Where [MODEL NAME] is the
                                  name attribute of :param build_model: and
                                  [VERSION NUMBER] is the version attribute of
                                  the hyperparameters object.
    :param checkpoint: bool, set to True to checkpoint the model during training
                       at /logs/models/[MODEL NAME]/version_[VERSION NUMBER]/
                       :param run_number:/checkpoints/
    :param log: bool, set to True to log metrics and losses in a csv file
                during training at /logs/models/[MODEL NAME]/
                version_[VERSION NUMBER]/:param run_number:/history.log

    ---> tensorflow.keras.Model, tensorflow.keras.History, a tuple of the trained
         model, along with its training history
    """

    # Building Model
    model, initial_epoch = build_compiled_model2(build_model, hparams,
                                                 metrics, run_number,
                                                 path_to_logs)
    model_name = model.name

    # Unpacking model training parameters
    ## Must deep copy the training parameters dictionary because otherwise var training_parameters is just pointer to
    ## the 'training_parameters' location of the hyperparameters object. This makes hyperparameters mutable,
    ## causing the hyperparameters to be locked during training because we add the mutable checkpointing callbacks
    ## this makes hyperparameters unpickleable on further reruns in in the same session.
    training_parameters = deepcopy(hparams['training_parameters'])

    # Checking if model has already completed training
    if (training_parameters['epochs'] - initial_epoch) == 0:
        return model, None

    if path_to_logs:
        # Setting up model checkpointing
        path_to_run = os.path.join(path_to_logs, 'models', model_name,
                                   '_'.join(['version',
                                             str(hparams['version'])]),
                                   'runs', str(run_number))
        path_to_ckpts = os.path.join(path_to_run, 'checkpoints')
        if not os.path.exists(path_to_ckpts):
            os.makedirs(path_to_ckpts)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(path_to_ckpts, 'cp-{epoch}.ckpt'), verbose=1, save_weights_only=True)
        # Setting up train logging
        csv_logger = tf.keras.callbacks.CSVLogger(filename=os.path.join(path_to_run, 'history.log'), append=True)
        if 'callbacks' not in training_parameters:
            training_parameters['callbacks'] = []
        training_parameters['callbacks'].append(cp_callback)
        training_parameters['callbacks'].append(csv_logger)
        # Writing Hyperparameters
        write_hparams(path_to_logs, model_name, hparams)

    # Training Model
    model_history = model.fit(X, y, **training_parameters,
                              initial_epoch=initial_epoch,
                              validation_data=validation_data)

    return model, model_history
