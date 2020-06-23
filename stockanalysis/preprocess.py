import os
import json
import numpy as np
import pandas as pd

from copy import deepcopy
from stockanalysis.text_normalization_methods import normalize_document

######################
## Helper Functions ##
######################

def append_vocab(document, file_path):
    """
    Adds to the already existing vocabulary file found at :param file_path: the
    new vocabulary found in the normalized document :param document:.

    :param document: string, normalized document to calculate vocabulary from.
    :param file_path: string, path to vocabulary json file

    ---> dict, vocab object, mapping words to their unique integer encodings
    """

    # Loading already established vocabulary
    try:
        with open(file_path, 'r') as f:
            vocab = json.load(f)

    except FileNotFoundError:
        vocab  = {}

    # Updating vocabulary dictionary
    if not vocab:
        last_word_encoding = 0
    else:
        last_word_encoding = max(vocab.values())

    for word in document.split():
        # if a word in the document is not in the current vocab, add it with a
        #  word encoding value larger than the largest word encoding value
        if word not in vocab:
            vocab[word] = last_word_encoding + 1
            last_word_encoding = last_word_encoding + 1

    with open(file_path, 'w') as f:
        json.dump(vocab, f)

    return vocab

##########################
## END Helper Functions ##
##########################


###########################################################
## Functions for preprocessing dataset pandas.DataFrames ##
###########################################################

def time_series_split(df, train_size=None, test_size=None):
    """
    Splits the dataset into train, validation, and test portions.

    :param df: pandas.DataFrame, representing our total dataset
    :param train_size: float, training set size
    :param test_size: float, test set size

    ---> pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, representing the train, validation, test datasets
    """

    dataset_size = len(df)

    if test_size != None:
        testset_index = int(test_size*dataset_size)
    elif train_size != None:
        testset_index = int((1 - train_size)*dataset_size)
    else:
        raise Exception('train_size and or test_size needs to be set')

    test_df = df.iloc[0:testset_index].copy(deep=True)
    train_df = df.iloc[testset_index:].copy(deep=True)

    return train_df, test_df

def window_df(df, columns, n_trail=1, n_lead=1):
    """
    :param df: DataFrame, dataframe object where the columns are the features
               and labels and the rows are days
    :param columns: list of strings, names of the features and labels
                    (columns of df) to be used in the time series
    :param n_trail: int, number of days behind day 0 that will be used to
                    predict days after day 0
    :param n_lead: int, number of days ahead of day 0 that will be predicted

    ---> pandas.DataFrame, dataframe object structured like a time series
         where each row represents an element in the time series, and each
         column is a feature or label a certain amount of days in the future
         or past.
    """

    df = df[columns]
    dfs = []
    col_names = []

    # Create trailing columns
    for i in range(n_trail, 0, -1):
        dfs.append(df.shift(-i))
        col_names += [(col_name + '(t-{})'.format(i)) for col_name in columns]

    # Create leading columns
    for i in range(0, n_lead+1):
        dfs.append(df.shift(i))
        col_names += [(col_name + '(t+{})'.format(i)) for col_name in columns]

    agg = pd.concat(dfs, axis=1)
    agg.columns = col_names

    agg.dropna(inplace=True)

    return agg

def extract_dataset(df, fcols, tcols, lag, forecast, single_step):
    """
    lag must > 0 same with forecast
    """

    w_df = window_df(df, set(fcols+tcols), n_trail=lag-1, n_lead=forecast)

    features = {fcol: w_df[
                           [col
                            for col in w_df.columns if fcol in col][:-forecast]
                           ].values
                for fcol in fcols}
    if single_step:
        targets = {'_'.join([tcol, 'target']): w_df[
                                                    [col
                                                     for col in w_df.columns
                                                     if tcol in col][-1]
                                                    ].values
                   for tcol in tcols}
    else:
        targets = {'_'.join([tcol, 'target']): w_df[
                                                    [col
                                                     for col in w_df.columns
                                                     if tcol in col][-forecast:]
                                                    ].values
                   for tcol in tcols}

    return features, targets

###############################################################
## END Functions for preprocessing dataset pandas.DataFrames ##
###############################################################


##########################################
## Functions for preprocessing datasets ##
##########################################

def sample_text_feature(feature, seed):
    """
    """

    def sample_element(el):
        texts = []
        for timestep in el:
            days_texts = json.loads(timestep)
            texts.extend(days_texts)
        if texts:
            text = np.random.choice(texts, size=1)[0]
        else:
            text = np.nan
        return text

    np.random.seed(seed)

    sampled_feature = np.asarray(list(map(sample_element, feature)))

    return sampled_feature


def norm_text_feature(feature, cut_off, norm_dir):
    """
    """

    def norm_text(link):
        print(link)
        if link == 'nan':
            norm_link = link
        else:
            root, doc_name = os.path.split(link)
            save_point = os.path.join(root, norm_dir)
            if not os.path.exists(save_point):
                os.makedirs(save_point)

            norm_link = os.path.join(save_point, doc_name)

            with open(link, 'r') as file:
                raw_document = file.read()
            norm_doc = normalize_document(raw_document,
                                          remove_large_words=cut_off)
            with open(norm_link, 'w') as norm_file:
                norm_file.write(norm_doc)
        return norm_link

    normed_feature = np.asarray(list(map(norm_text, feature)))

    return normed_feature

def encode_text_feature(feature, vocab):
    """
    """

    def encode_text(link):
        if link == 'nan':
            encoded_text = []
        else:
            with open(link, 'r') as f:
                text = f.read()
            encoded_text = [vocab.get(word, 0) for word in text.split()]
        return np.asarray(encoded_text)

    encoded_feature = np.asarray(list(map(encode_text, feature)))

    return encoded_feature

def pad_text_feature(feature):
    """
    """

    doc_lens = map(lambda arr: arr.shape[-1], feature)
    longest_doc_len = max(doc_lens)
    pad_doc = lambda arr: np.pad(arr, ((0, longest_doc_len - arr.shape[-1])),
                                 constant_values=0)
    return np.stack(list(map(pad_doc, feature)), axis=0)

def transform_ds(dataset, fnames, func, **params):
    """
    """

    features, targets = deepcopy(dataset)

    for fname in fnames:
        features[fname] = func(features[fname], **params)

    return features, targets

def shuffle_dataset(dataset, seed):
    """
    """

    np.random.seed(seed)
    features, labels = dataset
    dataset_size = len(next(iter(features.values())))
    shuffled_indices = np.random.choice(dataset_size, size=dataset_size,
                                        replace=False)
    features_shuffled = {fname: feature[shuffled_indices]
                         for fname, feature in features.items()}
    labels_shuffled = {lname: label[shuffled_indices]
                       for lname, label in labels.items()}
    return features_shuffled, labels_shuffled

# Functions for building vocabularies from datasets

def gen_feature_vocab(feature, path_to_vocab):
    """
    """

    def text_vocab(link):
        if link != 'nan':
            with open(link, 'r') as f:
                text = f.read()
            vocab = append_vocab(text, path_to_vocab)
        return None

    for link in feature:
        text_vocab(link)

    return None

def gen_vocabulary(dataset, fnames, path_to_vocab):
    """
    """

    features, targets = dataset

    for fname in fnames:
        gen_feature_vocab(features[fname], path_to_vocab)

    return None

def build_vocabulary(dataset, fnames, path_to_vocab):
    """
    """

    if os.path.isfile(path_to_vocab):
        os.remove(path_to_vocab)
    gen_vocabulary(dataset, fnames, path_to_vocab)
    with open(path_to_vocab, 'r') as f:
        vocab = json.load(f)

    return vocab


# END functions for building vocabularies from datasets

# Helper postprocessing functions

def encode_pad_dataset(dataset, fnames, vocab):
    """
    """

    encoded_ds = transform_ds(dataset, fnames, encode_text_feature, vocab=vocab)
    padded_ds = transform_ds(encoded_ds, fnames, pad_text_feature)

    return padded_ds

# END helper postprocessing functions

##############################################
## END Functions for preprocessing datasets ##
##############################################


#############################
## Preprocessing functions ##
#############################

def preprocess(df, feature_tickers, target_tickers, feature_names, target_names,
               lag, forecast, single_step, **kwargs):
    """
    """

    fcols = ['_'.join([n, t]) for n in feature_names for t in feature_tickers]
    tcols = ['_'.join([n, t]) for n in target_names for t in target_tickers]

    ds = extract_dataset(df, fcols, tcols, lag, forecast, single_step)

    if '8-k' in feature_names:
        # Getting text preprocessing parameters
        norm_dir = kwargs.get('norm_dir', 'norm')
        seed = kwargs.get('seed', None)
        try:
            cut_off = kwargs['cut_off']
        except KeyError as e:
            msg = 'preprocess() missing key word argument: \'cut_off\'. Argument is required when \'feature_names\' includes \'8-k\' as a feature to preprocess.'
            raise TypeError(msg) from e

        fcols_8k = ['_'.join(['8-k', t]) for t in feature_tickers]

        # Downsampling 8-K features to 1 document per sample
        ds = transform_ds(ds, fcols_8k, sample_text_feature, seed=seed)
        # Normalizing all text documents in the dataset
        ds = transform_ds(ds, fcols_8k, norm_text_feature, cut_off=cut_off,
                          norm_dir=norm_dir)
    return ds

#################################
## END Preprocessing functions ##
#################################
