"""
TODO:

1.) Decide and implement how preprocessing pipeline for model 1 will act when
    multiple text features are used.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from functools import reduce
from stockanalysis.text_normalization_methods import normalize_document

######################################################
## Functions for preprocessing pipeline for model 1 ##
######################################################

def normalize_text_features(df, fname, cut_off, tickers,
                            norm_docs_fname='normalized'):
    """
    Normalizes all documents linked to in df

    :param df: pandas.DataFrame, storing links and data
    :param fname: string, name of text feature to normalize
    :param cut_off: int, the cut off used when normalizing documents
    :param tickers: stock tickers to normalize documents for
    :param norm_docs_fname: string, filename for the file normalized documents
                            of a specific ticker will be stored in

    ---> pandas.DataFrame, with links updated to point to the normalized
         documents
    """

    def norm_doclist(doclist_string):
        """
        Takes a json formated string :param s: which contains a list of raw
        document paths, normalizes each document in the list, and returns an
        updated json formated string containing a list of links pointing to
        the normalized documents.

        :param doclist_string: string, json formated, when loaded contains a
                               list of paths to raw documents

        ---> string, json formatted, updated list of paths to normalized
             documents
        """

        def normalize_save_document(link):
            """
            Normalize, and save normalized document located at link. The
            normalized document is saved at location endpoint. Returns a path
            to the saved normalized document.

            :param link: string, path to document to be normalized
            :param endpoint: string, path to location to save
                             normalized document

            ---> string, path to saved normalized document
            """

            root, doc_name = os.path.split(link)

            save_point = os.path.join(root, norm_docs_fname)
            if not os.path.exists(save_point):
                os.makedirs(save_point)

            with open(link, 'r') as file:
                raw_document = file.read()

            norm_doc = normalize_document(raw_document,
                                          remove_large_words=cut_off)

            with open(os.path.join(save_point, doc_name), 'w') as norm_file:
                norm_file.write(norm_doc)

            return os.path.join(save_point, doc_name)


        return json.dumps(list(map(normalize_save_document, json.loads(doclist_string))))


    df_n = df.copy(deep=True)

    for t in tickers:
        df_n['_'.join([fname, t])] = df_n['_'.join([fname, t])].map(norm_doclist)

    return df_n

def calculate_log_returns(df, tickers, name='log_adj_daily_returns'):
    """
    Calculates the log adjusted returns feature from the adjusted closing
    price feature for each stock ticker in tickers.

    :param df: pandas.DataFrame
    :param tickers: stock tickers to calculate log adjusted returns for
    :param name: string, column name for the calculated feature

    ---> pandas.DataFrame, with log adjusted returns calculated
    """

    df = df.copy(deep=True)

    for t in tickers:
        df['_'.join(['log_adj_close', t])] = np.log(df['_'.join(['adjusted_close', t])])
        df['_'.join([name, t])] = df['_'.join(['log_adj_close', t])] - df['_'.join(['log_adj_close', t])].shift(-1)

    df = df.dropna(subset=['_'.join([name, tickers[0]])])

    return df


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

def build_vocabulary(df, fname, tickers, path_to_vocab):
    """
    Builds the vocabulary.json, from the corpus of documents related to each
    stock ticker.

    :param df: pandas.DataFrame
    :param fname: string, text feature to build vocabulary from
    :param tickers: list of strings, stock tickers to build the vocabulary off
                    of
    :param path_to_vocab: string, path to vocabulary json file

    ---> dict, representing the mapping between unique words and integers
    """

    def vocab_from_doclist(doclist_string):
        """
        Takes a json formated string :param s: that contains a list of
        document paths, and constructs or adds to an already existing
        vocab.json file, the unique words found in the documents refrenced in
        the list of document paths.

        :param doclist_string: string, json formated, contains a list of
                               paths to documents
        :param path_to_vocab: string, path to vocab.json file

        ---> :param doclist_string:
        """

        doclist = json.loads(doclist_string)

        for docpath in doclist:
            with open(docpath, 'r') as f:
                doc = f.read()

            vocab = append_vocab(doc, path_to_vocab)

        return json.dumps(doclist)


    df_v = df.copy(deep=True)

    for t in tickers:
        df_v['_'.join([fname, t])].map(vocab_from_doclist)

    with open(path_to_vocab, 'r') as f:
        vocab = json.load(f)

    return vocab

def encode_text_features(df, fname, tickers, vocab,
                         encode_docs_fname='encoded'):
    """
    Encodes all the text documents according to :param vocab: mapping

    :param df: pandas.DataFrame
    :param fname: string, text feature to encode according to :param vocab:
    :param tickers: list, list of stock tickers to use documents from
    :param vocab: dict, python dictionary mapping words to unique
                  integer encodings

    ---> pandas.DataFrame, with links updated to point to encoded documents
    """

    def encode_doclist(doclist_string):
        """
        Takes a json formated string :param s: which contains a list of document paths, encodes each document
        in the list according to :param vocab: and saves it as a pickle in a file named encoded, and
        returns an updated json formated string containing the list of encoded document paths.

        :param s: string, json formated, when loaded contains a list of paths to documents to encode
        :param vocab: dict, mapping individual words in documents to another python object that encodes an
                      individual word

        ---> string, json formated updated list of paths to encoded document files
        """

        def encode_save_document(link):
            """
            Takes a path to a document and encodes it according to outside state vocab, then saves encoded document
            as a pickle in a file named encoded.

            :param docpath: string, path to document to encode

            ---> string, path to encoded document pickle
            """

            root, doc_name = os.path.split(link)

            # Defining save point for encoded documents
            save_point = os.path.join(root, encode_docs_fname)
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            with open(link, 'r') as tfile:
                text = tfile.read()

            # Encoding document
            encoded_document = [vocab[word] for word in text.split()]

            # Saving encoded document
            doc_name = '.'.join([os.path.splitext(doc_name)[0], 'pickle'])
            with open(os.path.join(save_point, doc_name), 'wb') as bfile:
                pickle.dump(encoded_document, bfile)

            return os.path.join(save_point, doc_name)


        return json.dumps(list(map(encode_save_document, json.loads(doclist_string))))


    df_e = df.copy(deep=True)

    for t in tickers:
        df_e['_'.join([fname, t])] = df_e['_'.join([fname, t])].map(encode_doclist)

    return df_e

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

# Functions for preprocessing datasets

def extract_features(feature_names, ticker, df):
    cols = map(lambda fname: '_'.join([fname, ticker]), feature_names)
    return df[cols]

def extract_window_dataset(df, feature_names, ticker, n_trail=1, n_lead=1):
    selected_df = extract_features(feature_names, ticker, df)
    windowed_df = window_df(selected_df, selected_df.columns,
                            n_trail=n_trail, n_lead=n_lead)
    dataset = {fname: windowed_df[filter(lambda name: True if fname in name else False, windowed_df.columns)].values
                     for fname in feature_names}
    return dataset

def extract_labels(dataset, label_names):
    labels = {'_'.join([fname, 'target']): dataset[fname][:, -1] for fname in dataset.keys() if fname in label_names}
    features = {fname: dataset[fname][:, :-1] for fname in dataset.keys()}
    return features, labels

def reshape_docs_feature(dataset, tfname, seed):

    def flatten_docs_feature(docs_feature):
        docs = []
        for timestep in docs_feature:
            docs_list = json.loads(timestep)
            docs.extend(docs_list)
        return docs

    def sample_docs(docs_feature, seed):
        np.random.seed(seed)
        docs_names = flatten_docs_feature(docs_feature)
        if len(docs_names) != 0:
            windows_doc_name = np.random.choice(docs_names, size=1)[0]
            with open(windows_doc_name, 'rb') as f:
                windows_doc = pickle.load(f)
            window = windows_doc
        else:
            window = []
        return np.asarray(window)

    features = dataset[0]
    labels = dataset[1]

    features_reshaped = {key: (value if key != tfname
                               else list(map(lambda docf: sample_docs(docf, seed), value)))
                         for key, value in features.items()}

    return features_reshaped, labels

def filter_dataset(dataset, tfname):
    features = dataset[0]
    labels = dataset[1]
    mask = [(sample.shape[0] != 0) for sample in features[tfname]]
    features_filtered = {key: (value[mask, :] if key != tfname
                               else [value[i] for i in range(len(mask)) if mask[i]])
                         for key, value in features.items()}
    labels_filtered = {key: value[mask] for key, value in labels.items()}
    return features_filtered, labels_filtered

def add_datasets(dataset_1, dataset_2, tfname):
    features_1 = dataset_1[0]
    labels_1 = dataset_1[1]
    features_2 = dataset_2[0]
    labels_2 = dataset_2[1]
    assert (features_1.keys() == features_2.keys()) and (labels_1.keys() == labels_2.keys())
    feature_names = features_1.keys()
    label_names = labels_1.keys()
    features = {fname: (np.concatenate((features_1[fname], features_2[fname]))
                        if fname != tfname else features_1[fname] + features_2[fname])
                for fname in feature_names}
    labels = {lname: np.concatenate((labels_1[lname], labels_2[lname]))
              for lname in label_names}
    return features, labels

def pad_documents(docs_feature):
    shapes = map(lambda arr: arr.shape, docs_feature)
    longest_doc_len = max(map(lambda shape: shape[-1], shapes))
    pad_doc = lambda arr:  np.pad(arr, ((0, longest_doc_len - arr.shape[-1])), constant_values=0)
    return np.stack(list(map(pad_doc, docs_feature)), axis=0)

def pad_dataset(dataset, tfname):
    features = dataset[0]
    labels = dataset[1]
    pad_document_feature = lambda feature_name, f: (feature_name, pad_documents(f)) if feature_name == tfname else (feature_name, f)
    padded_features = dict(map(lambda item: pad_document_feature(item[0], item[1]), features.items()))
    return padded_features, labels

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    features = dataset[0]
    labels = dataset[1]
    dataset_size = len(next(iter(features.values())))
    shuffled_indices = np.random.choice(dataset_size, size=dataset_size, replace=False)
    features_shuffled = {fname: feature[shuffled_indices] for fname, feature in features.items()}
    labels_shuffled = {lname: label[shuffled_indices] for lname, label in labels.items()}
    return features_shuffled, labels_shuffled

# END Functions for preprocessing datasets

# Preprocessing Function

def preprocess(df, params, tickers, seed=None):
    """
    Preprocesses data. This includes normalizing documents of the given stock
    tickers. Generating a vocabulary json object representing the mapping
    between unique words and their integer encodings for the corpus of
    documents normalized. Encoding the normalized documents according to said
    vocabulary. Calculatating the adjusted logarithmic returns for each given
    stock ticker.The data is segmented by stock tickker, then windowed into
    the frame of 6 day intervals. The 6th day is then extracted as the target
    variable for the preceding 5 day interval. A single document within each
    interval is uniformly selected to be the text feature for the specific 5
    day interval. The data for each stock ticker is then merged back togther
    to form a single dataset. The text features this dataset are then padded
    to the same length with 0 characters (since our vocabulary starts at 1)
    and the dataset is shuffled.

    :param df: pandas.DataFrame used to house gathered data
    :param params: dict, containing the parameters used for preprocessing.
                   keys: 'cut_off': (the cut off in normalize_document),
                         'window_size': (the size of the interval of previous
                                        features used to predict the current
                                        value),
                         'norm_docs_fname': (the name of the file to store
                                             normalized documents in),
                         'encode_docs_fname': (the name of the file to store
                                               encoded documents in),
                         'path_to_vocab': (path to the vocabulary json file)
    :param tickers: list of strings, where each string is a stock ticker
    :param seed: int, random seed

    ---> ((features, labels), vocab), where features is a dict with keys for each
         feature, and values with length equal to the sample size of the
         dataset. Similarly labels is a dict with keys for each target label
         and values with lenght equal to the sample size of the dataset. Vocab
         is the dict mapping integers to the words they represent.
    """

    # Preprocessing Parameters
    log_adj_ret_name = params['log_adj_ret_name']
    feature_names = params['feature_names']
    label_feature_names = params['label_feature_names']
    text_feature_names = params['text_feature_names']
    cut_off = params['cut_off']
    window_size = params['window_size']
    norm_docs_fname = params['norm_docs_fname']
    path_to_vocab = params['path_to_vocab']
    encode_docs_fname = params['encode_docs_fname']

    df = normalize_text_features(df, text_feature_names, cut_off, tickers,
                                 norm_docs_fname)
    vocab = build_vocabulary(df, text_feature_names, tickers,
                             path_to_vocab)
    df = encode_text_features(df, text_feature_names, tickers, vocab,
                              encode_docs_fname)
    df = calculate_log_returns(df, tickers, name=log_adj_ret_name)
    datasets = map(lambda t: extract_window_dataset(df, feature_names,
                                                    ticker=t,
                                                    n_trail=window_size-1),
                   tickers)
    datasets = map(lambda ds: extract_labels(ds, label_feature_names), datasets)
    datasets = map(lambda ds: reshape_docs_feature(ds, text_feature_names,
                                                   seed), datasets)
    datasets = map(lambda ds: filter_dataset(ds, text_feature_names), datasets)
    dataset = reduce(lambda ds1, ds2: add_datasets(ds1, ds2, text_feature_names),
                     datasets)
    dataset = pad_dataset(dataset, text_feature_names)
    dataset = shuffle_dataset(dataset, seed)

    return dataset, vocab

##########################################################
## END Functions for preprocessing pipeline for model 1 ##
##########################################################
