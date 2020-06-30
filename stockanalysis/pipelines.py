import os
import shutil
import json
import pickle
import numpy as np
import tensorflow as tf
import stockanalysis.data as d
import stockanalysis.models as m

from copy import deepcopy
from stockanalysis.train import train2, config_hardware
from stockanalysis.preprocess import preprocess, time_series_split, build_vocabulary, encode_pad_dataset, shuffle_dataset

##################################################################
## Functions for accessing metadata storage, and model registry ##
##################################################################

class MetaDataStore:

    run_meta_fname = 'run_{}.json'
    run_meta_default = {
                        'states': [('start', {})],
                        'meta': {'deployed_model': None, 'run_metrics': None}
                        }

    def __init__(self, path_to_metadata, path_to_data, path_to_models,
                 pipeline_config):
        self.path = path_to_metadata
        # Setting up file structure
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        meta_file = os.path.join(self.path, 'meta.json')
        if os.path.isfile(meta_file):
            with open(meta_file, 'r') as f:
                cur_meta = json.load(f)
                path_to_config = cur_meta['config']
                with open(path_to_config, 'rb') as f:
                    cur_config = pickle.load(f)
                cur_path_to_data = cur_meta['path_to_data']
                cur_path_to_models = cur_meta['path_to_models']
                if ((cur_config != pipeline_config) or (cur_path_to_data != path_to_data) or (cur_path_to_models != path_to_models)):
                    msg = 'Previous pipeline config does not match current configuration'
                    raise ValueError(msg)
        else:
            meta = {'run_number': 0, 'path_to_data': path_to_data, 'path_to_models': path_to_models, 'config': os.path.join(self.path, 'config.pickle')}
            with open(meta_file, 'w') as f:
                json.dump(meta, f)
            with open(os.path.join(self.path, 'config.pickle'), 'wb') as f:
                pickle.dump(pipeline_config, f)
            # Cleaning up already existing paths to data and models
            if os.path.isdir(path_to_data) and (len(os.listdir(path_to_data)) > 0):
                shutil.rmtree(path_to_data)
            if os.path.isdir(path_to_models) and (len(os.listdir(path_to_models)) > 0):
                shutil.rmtree(path_to_models)
            if not os.path.isdir(path_to_data):
                os.makedirs(path_to_data)
            if not os.path.isdir(path_to_models):
                os.makedirs(path_to_models)
        run_meta_file = os.path.join(self.path, self.run_meta_fname.format(0))
        if not os.path.isfile(run_meta_file):
            with open(run_meta_file, 'w') as f:
                json.dump(self.run_meta_default, f)

    def get_run_number(self):
        meta_file = os.path.join(self.path, 'meta.json')
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        return meta['run_number']

    def set_run_number(self, num):
        meta_file = os.path.join(self.path, 'meta.json')
        # Setting new run number
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        meta['run_number'] = num
        with open(meta_file, 'w') as f:
            json.dump(meta, f)
        # Dumping new run metadata file
        new_run_file = os.path.join(self.path, self.run_meta_fname.format(num))
        with open(new_run_file, 'w') as f:
            json.dump(self.run_meta_default, f)

    def get_pipeline_state(self):
        run_number = self.get_run_number()
        run_file = os.path.join(self.path,
                                self.run_meta_fname.format(run_number))
        with open(run_file, 'r') as f:
            run_metadata = json.load(f)
        states = run_metadata['states']
        cur_state = states[-1]
        return cur_state

    def set_pipeline_state(self, state, **params):
        run_number = self.get_run_number()
        run_file = os.path.join(self.path,
                                self.run_meta_fname.format(run_number))
        with open(run_file, 'r') as f:
            run_metadata = json.load(f)
        run_metadata['states'].append((state, params))
        with open(run_file, 'w') as f:
            json.dump(run_metadata, f)

    def write_run_metadata(self, deployed_model, run_metrics):
        run_number = self.get_run_number()
        run_file = os.path.join(self.path,
                                self.run_meta_fname.format(run_number))
        with open(run_file, 'r') as f:
            run_metadata = json.load(f)
        run_metadata['meta']['deployed_model'] = deployed_model
        run_metadata['meta']['run_metrics'] = run_metrics
        with open(run_file, 'w') as f:
            json.dump(run_metadata, f)

class ModelStore:

    def __init__(self, path_to_models):
        self.path = path_to_models
        # Setting up paths and file structure
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.testing = os.path.join(self.path, 'testing')
        if not os.path.isdir(self.testing):
            os.mkdir(self.testing)
        self.deployed = os.path.join(self.path, 'deployed_model')
        if not os.path.isdir(self.deployed):
            os.mkdir(self.deployed)

    @staticmethod
    def contains_model(path):
        return os.path.exists(os.path.join(path, 'saved_model.pb'))

    def get_current_model_version(self):
        versions = [int(dir) for dir in os.listdir(self.deployed)
                    if self.contains_model(os.path.join(self.deployed, dir))
                    and dir.isdigit()]
        if versions:
            cur_ver = max(versions)
        else:
            cur_ver = None
        return cur_ver

    def extract_model(self, type, gpu_memory):
        config_hardware(gpu_memory, seed=None)
        if type == 'testing':
            if self.contains_model(self.testing):
                model = tf.keras.models.load_model(self.testing, custom_objects={'tf': tf})
            else:
                model = None
            if os.path.isfile(os.path.join(self.testing, 'vocab.json')):
                with open(os.path.join(self.testing, 'vocab.json'), 'r') as f:
                    vocab = json.load(f)
            else:
                vocab = None
        elif type == 'deployed':
            cv = self.get_current_model_version()
            if cv is not None:
                model = tf.keras.models.load_model(os.path.join(self.deployed,
                                                                str(cv)),
                                                                custom_objects={'tf': tf})
            else:
                model = None
            if os.path.isfile(os.path.join(self.deployed, str(cv),
                                           'vocab.json')):
                with open(os.path.join(self.deployed, str(cv),
                                       'vocab.json'), 'r') as f:
                    vocab = json.load(f)
            else:
                vocab = None
        else:
            raise ValueError
        return model, vocab

    def load_model(self, model, vocab, type):
        if type == 'testing':
            model.save(self.testing)
            if vocab is not None:
                with open(os.path.join(self.testing, 'vocab.json'), 'w') as f:
                    json.dump(vocab, f)
        elif type == 'deployed':
            cur_ver = self.get_current_model_version()
            if cur_ver is not None:
                m_ver = cur_ver + 1
            else:
                m_ver = 0
            m_path = os.path.join(self.deployed, str(m_ver))
            model.save(m_path)
            if vocab is not None:
                with open(os.path.join(m_path, 'vocab.json'), 'w') as f:
                    json.dump(vocab, f)
        else:
            raise ValueError

class DataStore:

    def __init__(self, path_to_data, tickers, source, form_types):
        self.path = path_to_data
        self.tickers = tickers
        self.source = source
        self.form_types = form_types
        # Setting up file structure
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.fresh = os.path.join(self.path, 'fresh')
        if not os.path.isdir(self.fresh):
            os.mkdir(self.fresh)
        self.deployed = os.path.join(self.path, 'deployed')
        if not os.path.isdir(self.deployed):
            os.mkdir(self.deployed)

    def update(self):
        d.fetch_data(self.fresh, self.tickers, self.source, self.form_types)

    def extract(self, type):
        if type == 'raw':
            data = d.load_data(self.fresh)
        elif (type == 'full') or (type == 'train') or (type == 'test'):
            dataset_path = os.path.join(self.fresh,
                                        '_'.join([type, 'dataset.pickle']))
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError
        return data

    def load(self, data, type):
        if (type == 'full') or (type == 'train') or (type == 'test'):
            dataset_path = os.path.join(self.fresh,
                                        '_'.join([type, 'dataset.pickle']))
            with open(dataset_path, 'wb') as f:
                pickle.dump(data, f, protocol=4)
        elif type == 'raw':
            raise NotImplementedError
        else:
            raise ValueError

    def depricate_fresh(self):
        if os.path.isdir(self.deployed):
            shutil.rmtree(self.deployed)
        shutil.copytree(self.fresh, self.deployed)

######################################################################
## END Functions for accessing metadata storage, and model registry ##
######################################################################

#####################
## Pipeline Stages ##
#####################

def val_data(fresh_data, config):
    """
    Determines whether :param fresh_data: is consumable by the default pipeline.

    :param fresh_data: data type required for val_data,

    ---> bool, True if data is consumable, False otherwise
    """

    # Checking Schema
    schema = d.get_feature_names(fresh_data)
    if config['model'] == 'model_text':
        m.val_schema_model_text(schema)
    elif config['model'] == 'baseline_model':
        m.val_schema_baseline_model(schema)
    else:
        raise RuntimeError('Model: {} not supported.'.format(config['model']))

    return True

def prep_data(fresh_data, config):
    """
    ---> (train_ds_path, test_ds_path), ds_path
    """

    # Splitting raw data
    train_size = 0.8

    train_data, test_data = time_series_split(fresh_data, train_size=train_size)

    # Preprocessing data
    ## Setting preprocessing parameters
    preprocess_params = config['preprocessing']

    ## Preprocessing train, test and full data splits
    train_dataset = preprocess(train_data, **preprocess_params)
    test_dataset = preprocess(test_data, **preprocess_params)
    dataset = preprocess(fresh_data, **preprocess_params)

    return (train_dataset, test_dataset), dataset

def train_model(train_dataset, config, deploy, metadata_store, gpu_memory,
                test_dataset=None):
    """
    ---> path_to_model
    """

    run_number = metadata_store.get_run_number()

    if config['model'] == 'model_text':
        # Setting build model
        build_model = m.model_text
        # Gathering 8-K Features
        fnames_8k = [fname for fname in train_dataset[0] if '8-k' in fname]
        # Building vocab (don't like the below fix, but I'd have to change how build vocab works...)
        temp_vocab_path = os.path.join(metadata_store.path, 'train_vocab.json')
        train_vocab = build_vocabulary(train_dataset, fnames_8k, temp_vocab_path)
        os.remove(temp_vocab_path)
        # Encoding, and padding the text features dataset
        train_dataset = encode_pad_dataset(train_dataset, fnames_8k,
                                           train_vocab)
        if test_dataset:
            test_dataset = encode_pad_dataset(test_dataset, fnames_8k,
                                              train_vocab)
        # Setting hyperparameters
        hparams = deepcopy(config['hyperparameters'])
        hparams['model_parameters']['vocab'] = train_vocab
        hparams['model_parameters']['output_bias'] = {
                                                      tname: train_dataset[1][tname].mean()
                                                      for tname in train_dataset[1]
                                                      }
        if deploy:
            hparams['version'] = 'final'
        else:
            hparams['version'] = 'testing'
        emb_dir = os.path.join(metadata_store.path, 'emb_dir')
        if not os.path.isdir(emb_dir):
            os.makedirs(emb_dir)
        emb_name = '_'.join(['emb', hparams['version'], '{}.pickle'])
        emb_path = os.path.join(emb_dir, emb_name)
        prev_emb_path = emb_path.format(run_number-1)
        if os.path.exists(prev_emb_path):
            os.remove(prev_emb_path)
        hparams['model_parameters']['emb_path'] = emb_path.format(run_number)
    elif config['model'] == 'baseline_model':
        train_vocab = None
        # Setting build model
        build_model = m.baseline_model
        # Setting hyperparameters
        hparams = deepcopy(config['hyperparameters'])
        hparams['model_parameters']['output_bias'] = {
                                                      tname: train_dataset[1][tname].mean()
                                                      for tname in train_dataset[1]
                                                      }
        if deploy:
            hparams['version'] = 'final'
        else:
            hparams['version'] = 'testing'
    else:
        raise RuntimeError('Model: {} not supported.'.format(config['model']))
    # Shuffling dataset
    X_train, y_train = shuffle_dataset(train_dataset, seed=None)
    if test_dataset:
        test_dataset = shuffle_dataset(test_dataset, seed=None)
    # Setting metrics
    metrics = deepcopy(config['metrics'])
    # Training model
    path_to_logs = os.path.join(metadata_store.path, 'logs')
    config_hardware(gpu_memory, seed=None)
    model, model_history = train2(build_model, hparams, metrics, run_number,
                                  X_train, y_train, test_dataset, path_to_logs)

    return model, train_vocab

def eval_model(model, test_dataset):
    """
    ---> eval_metrics
    """

    # Loading trained model
    model, train_vocab = model

    if model.name == 'model_text':
        # Encoding the test dataset according to the vocab learned from training
        fnames_8k = [fname for fname in test_dataset[0] if '8-k' in fname]
        test_dataset = encode_pad_dataset(test_dataset, fnames_8k, train_vocab)
    elif model.name != 'baseline_model':
        raise RuntimeError('Model: {} not supported.'.format(model.name))

    X_test, y_test = shuffle_dataset(test_dataset, seed=None)

    # Evaluating model REFACTOR HERE
    ## Calculating built in model metrics and losses
    metrics_test_raw = model.evaluate(X_test, y_test, batch_size=1)
    metrics_test_raw = [] + (metrics_test_raw if type(metrics_test_raw) == list
                             else [metrics_test_raw])
    metrics_test = dict(zip(model.metrics_names, metrics_test_raw))
    ## Calculating model's daily trend predictions on the test dataset
    preds_test = model.predict(X_test, batch_size=1)
    ## Calculating the test dataset's daily trends
    y_trend_test = {'_'.join(['daily_trend']+tname.split('_')[-2:]): ((y_test[tname] - X_test['_'.join(tname.split('_')[:-1])][:, -1]) > 0).astype(int) for tname in y_test}
    preds_trend_test = {'_'.join(['daily_trend']+tname.split('_')[-2:]): ((preds_test[tname] - X_test['_'.join(tname.split('_')[:-1])][:, -1]) > 0).astype(int) for tname in preds_test}
    ## Calculating the model's accuracy at predicting daily trends for the test dataset
    trend_accs_test = {'_'.join(tname.split('_')[:-1]+['acc']): np.mean(np.equal(y_trend_test[tname], preds_trend_test[tname])) for tname in y_trend_test}
    ## Combining both the built in metrics with the manually calculated metrics
    metrics_test.update(trend_accs_test)

    return metrics_test

def val_model(eval_metrics, test_dataset, model_store, gpu_memory):
    """
    ---> val_metrics (combo of both val metrics for documentation), retrain (bool)
    """

    model, vocab = model_store.extract_model('deployed', gpu_memory)

    if model is None:
        retrain = True
        val_metrics = {'testing': eval_metrics, 'deployed': None}
    else:
        if model.name == 'model_text':
            # Encoding the test dataset according to the deployed model's vocab
            fnames_8k = [fname for fname in test_dataset[0] if '8-k' in fname]
            test_dataset = encode_pad_dataset(test_dataset, fnames_8k, vocab)
        elif model.name != 'baseline_model':
            raise RuntimeError('Deployed model: {} not supported.'.format(model.name))

        X_test, y_test = shuffle_dataset(test_dataset, seed=None)

        # Evaluating model REFACTOR HERE
        ## Calculating built in model metrics and losses
        metrics_test_raw = model.evaluate(X_test, y_test, batch_size=1)
        metrics_test_raw = [] + (metrics_test_raw if type(metrics_test_raw) == list
                                 else [metrics_test_raw])
        metrics_test = dict(zip(model.metrics_names, metrics_test_raw))
        ## Calculating model's daily trend predictions on the test dataset
        preds_test = model.predict(X_test, batch_size=1)
        ## Calculating the test dataset's daily trends
        y_trend_test = {'_'.join(['daily_trend']+tname.split('_')[-2:]): ((y_test[tname] - X_test['_'.join(tname.split('_')[:-1])][:, -1]) > 0).astype(int) for tname in y_test}
        preds_trend_test = {'_'.join(['daily_trend']+tname.split('_')[-2:]): ((preds_test[tname] - X_test['_'.join(tname.split('_')[:-1])][:, -1]) > 0).astype(int) for tname in preds_test}
        ## Calculating the model's accuracy at predicting daily trends for the test dataset
        trend_accs_test = {'_'.join(tname.split('_')[:-1]+['acc']): np.mean(np.equal(y_trend_test[tname], preds_trend_test[tname])) for tname in y_trend_test}
        ## Combining both the built in metrics with the manually calculated metrics
        metrics_test.update(trend_accs_test)
        # If all metrics are better then retrain is set to True
        retrain = all((eval_metrics[m_name] < metrics_test[m_name])
                      if 'loss' in m_name else
                      (eval_metrics[m_name] > metrics_test[m_name])
                      for m_name in metrics_test)
        val_metrics = {'testing': eval_metrics, 'deployed': metrics_test}

    return val_metrics, retrain

#########################
## END Pipeline Stages ##
#########################

#############################
## Pipeline Implementation ##
############################

def pipeline(path_to_metadata, path_to_data, path_to_models,
             gpu_memory, config):
    """
    """
    metadata_store = MetaDataStore(path_to_metadata, path_to_data,
                                   path_to_models, config)
    model_store = ModelStore(path_to_models)
    data_store = DataStore(path_to_data, **config['data_info'])

    state, params = metadata_store.get_pipeline_state()

    # Updating data store
    if state == 'start':
        print('updating data store')
        data_store.update()
        state = 'val_data'
        metadata_store.set_pipeline_state(state)
    # Validating fresh data
    if state == 'val_data':
        print('validating fresh data')
        fresh_data = data_store.extract('raw')
        if val_data(fresh_data, config):
            state = 'prep_data'
            metadata_store.set_pipeline_state(state)
        else:
            cv = model_store.get_current_model_version()
            if cv is not None:
                path_to_dep_model = os.path.join(model_store.deployed, str(cv))
            else:
                path_to_dep_model = None
            val_metrics = None
            metadata_store.write_run_metadata(path_to_dep_model, val_metrics)
            ## Maybe raise an message to the DS team that the most recent data didnt pass validation
            state = 'completed'
            metadata_store.set_pipeline_state(state)
    # Preparing data
    if state == 'prep_data':
        print('preparing fresh data')
        fresh_data = data_store.extract('raw')
        (train_data, test_data), data = prep_data(fresh_data, config)
        data_store.load(train_data, 'train')
        data_store.load(test_data, 'test')
        data_store.load(data, 'full')
        state = 'train_model'
        metadata_store.set_pipeline_state(state)
    # Training model
    if state == 'train_model':
        print('training model on fresh data')
        train_dataset = data_store.extract('train')
        test_dataset = data_store.extract('test')
        model, vocab = train_model(train_dataset, config, False, metadata_store,
                                   gpu_memory, test_dataset)
        model_store.load_model(model, vocab, 'testing')
        state = 'eval_model'
        metadata_store.set_pipeline_state(state)
    # Evaluating model
    if state == 'eval_model':
        print('evaluating model on fresh data')
        test_dataset = data_store.extract('test')
        model = model_store.extract_model('testing', gpu_memory)
        eval_metrics = eval_model(model, test_dataset)
        state = 'val_model'
        params = {'eval_metrics': eval_metrics}
        metadata_store.set_pipeline_state(state, **params)
    # Validating models
    if state == 'val_model':
        print('validating deployed model')
        test_dataset = data_store.extract('test')
        eval_metrics = params['eval_metrics']
        val_metrics, retrain = val_model(eval_metrics, test_dataset,
                                         model_store, gpu_memory)
        state = 'deploy_model'
        params = {'val_metrics': val_metrics, 'retrain': retrain}
        metadata_store.set_pipeline_state(state, **params)
    # Deploying model
    if state == 'deploy_model':
        retrain = params['retrain']
        val_metrics = params['val_metrics']
        if retrain:
            print('Retraining and deploying model')
            dataset = data_store.extract('full')
            model, vocab = train_model(dataset, config, True, metadata_store,
                                       gpu_memory)
            model_store.load_model(model, vocab, 'deployed')
            data_store.depricate_fresh()
        cv = model_store.get_current_model_version()
        if cv is not None:
            path_to_dep_model = os.path.join(model_store.deployed, str(cv))
        else:
            path_to_dep_model = None
        metadata_store.write_run_metadata(path_to_dep_model, val_metrics)
        state = 'completed'
        metadata_store.set_pipeline_state(state)
    # Increasing run number if the run
    if state == 'completed':
        next_run = metadata_store.get_run_number() + 1
        metadata_store.set_run_number(next_run)

    return None

#################################
## END Pipeline Implementation ##
#################################
