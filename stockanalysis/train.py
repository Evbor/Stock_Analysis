import os
import tensorflow as tf

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
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    # Setting Tensorflow Seed
    tf.random.set_seed(seed)

    return None

def build_compiled_model(build_model, hparams, metrics, run_number):
    """
    Builds compiled model from build model function.

    :param build_model: function, the function that returns a uncompiled model
    :param hparams: dict, key word indexed hyperparamters for constructing the
                    model defined by :param build_model:
    :param metrics: tensorflow.metrics, to compile the model with
    :param run_number: int, run number uniquely identifying the returned
                       compiled model, used loading a model from a saved checkpoint

    ---> tensorflow.keras.Model, that is compiled and ready to fit on a dataset
         or predict
    """
    model_name = build_model.__name__
    hparam_version = hparams['version']
    loss = hparams['loss']
    optimizer = hparams['optimizer']
    model_parameters = hparams['model_parameters']
    model = build_model(**model_parameters)
    path_to_ckpt = os.path.join('logs', 'models', model_name, '_'.join(['version', str(hparam_version)]),
                               'runs', str(run_number), 'checkpoints')
    if os.path.exists(path_to_ckpt):
        latest = tf.train.latest_checkpoint(path_to_ckpt)
        model.load_weights(latest)
        print('Restored model from: {}'.format(latest))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model



def train(build_model, hparams, metrics, run_number, X, y, gpu_memory, seed):
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
                    train models on
    :param gpu_memory: int >= 0 or None, amount of GPU memory to allocate for
                       training models
    :param seed: int, random seed to define for training models

    ---> tensorflow.keras.Model, tensorflow.keras.History, a tuple of the trained
         model, along with its training history
    """

    config_hardware(gpu_memory, seed)

    training_parameters = hparams['training_parameters']
    model = build_compiled_model(build_model, hparams, metrics, run_number)
    model_history = model.fit(X, y, **training_parameters)

    return model, model_history
