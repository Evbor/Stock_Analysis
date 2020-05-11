import os
import pickle
import tensorflow as tf

#######################
## Model Definitions ##
#######################

# Default Model Definitions

def embedding_matrix(vocab, init, emb_name):
    '''
    Constructs the embedding matrix for specific init type for a pre-initialized word embedding layer.

    :param vocab: dict, a mapping between keys of words, and values of unique integer identifiers for each word
    :param init: string, initialization type currently we only support glove initialization

    ---> numpy array of size (vocab length, embedding dimension) mapping each word encoding to a vector
    '''

    if init == 'glove':
        glove_dir = os.path.join('~', '.stockanalysis', 'model_resources', 'glove')
        glove_dir = os.path.expanduser(glove_dir)

        try:
            with open(os.path.join(glove_dir, '{}.pickle'.format(emb_name)), 'rb') as f:
                embedding_m = pickle.load(f)

        except FileNotFoundError:
            # Building word to vector map
            word_embeddings = {}
            with open(os.path.join(glove_dir, 'glove.840B.300d.txt')) as f:
                for line in f:
                    tokens = line.split(' ')
                    word = tokens[0]
                    embedding = np.asarray(tokens[1:], dtype='float32')
                    # Needs to check if dim is changing
                    assert len(embedding) == 300
                    word_embeddings[word] = embedding
            # Building embedding matrix
            EMBEDDING_DIM = len(next(iter(word_embeddings.values())))
            embedding_m = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
            for word, i in vocab.items():
                embedding_vector = word_embeddings.get(word)
                if embedding_vector is not None:
                    embedding_m[i] = embedding_vector
            # Saving embedding matrix
            with open(os.path.join(glove_dir, '{}.pickle'.format(emb_name)), 'wb') as f:
                pickle.dump(embedding_m, f)

    else:
        raise ValueError('init type not supported, init must be equal to "glove"')

    return embedding_m

def Word_Embedding(vocab, init, emb_name,
                   embeddings_initializer='uniform', embeddings_regularizer=None,
                   activity_regularizer=None, embeddings_constraint=None,
                   mask_zero=False, input_length=None, **kwargs):

    '''
    Creates a keras embedding layer specifically designed to embed the words specified in :param vocab:

    :param vocab: dict, representing the mapping between the words in corpus (keys) and their unique integer
                  encodings
    :param init: string or int, tells the layer how to initialize its embeddings. If of type int, then
                 it tells the layer to initialize its word embeddings with an embedding dimension of :param init:.
                 If of type string, then :param init: specifies the type of pretrained word embeddings we will be
                 initializing the embedding layer with

    ---> tf.keras.layers.Embedding
    '''

    if isinstance(init, str):
        current_embedding_matrix = embedding_matrix(vocab, init, emb_name)
        emb_layer = tf.keras.layers.Embedding(current_embedding_matrix.shape[0], current_embedding_matrix.shape[1],
                                              weights=[current_embedding_matrix], mask_zero=mask_zero,
                                              input_length=None, **kwargs)

    elif isinstance(init, int):
        emb_layer = tf.keras.layers.Embedding(len(vocab) + 1, output_dim=init,
                                              embeddings_initializer=embeddings_initializer, embeddings_regularizer=embeddings_regularizer,
                                              activity_regularizer=activity_regularizer, embeddings_constraint=embeddings_constraint,
                                              mask_zero=mask_zero, input_length=input_length, **kwargs)
    else:
        raise ValueError('init type not supported')

    return emb_layer

def document_embedder_model(vocab, emb_name, doc_embedding_size):
    input_doc = tf.keras.Input(shape=(None,), name='doc')
    word_embedding = Word_Embedding(vocab, init='glove', emb_name=emb_name, mask_zero=False, trainable=False)(input_doc)
    document_embedding = tf.keras.layers.LSTM(doc_embedding_size)(word_embedding)
    model = tf.keras.Model(input_doc, document_embedding, name='document_embedder')
    return model

def model_0(vocab, lstm_layer_units=32, doc_embedding_size=100, emb_name='current_embedding',
            output_kernel_init=None, output_bias_init=None):

    if output_bias_init is not None:
        output_bias_init = tf.keras.initializers.Constant(output_bias_init)

    inputs = {
              'adjusted_close_WFC': tf.keras.Input(shape=(5,), name='adjusted_close_WFC', dtype=tf.float32),
              '8-k_WFC': tf.keras.Input(shape=(None,), name='8-k_WFC', dtype=tf.int64),
              'adjusted_close_JPM': tf.keras.Input(shape=(5,), name='adjusted_close_JPM', dtype=tf.float32),
              '8-k_JPM': tf.keras.Input(shape=(None,), name='8-k_JPM', dtype=tf.int64),
              'adjusted_close_BAC': tf.keras.Input(shape=(5,), name='adjusted_close_BAC', dtype=tf.float32),
              '8-k_BAC': tf.keras.Input(shape=(None,), name='8-k_BAC', dtype=tf.int64),
              'adjusted_close_C': tf.keras.Input(shape=(5,), name='adjusted_close_C', dtype=tf.float32),
              '8-k_C': tf.keras.Input(shape=(None,), name='8-k_C', dtype=tf.int64),
             }

    doc_embedder = document_embedder_model(vocab, emb_name, doc_embedding_size)
    document_embeddings = [doc_embedder(inputs[fname]) for fname in inputs.keys() if '8-k' in fname]

    reshape_doc_embedding = tf.keras.layers.Lambda(lambda x: tf.keras.backend.stack([x for i in range(5)], axis=1))
    reshaped_doc_embeddings = [reshape_doc_embedding(doc_embedding) for doc_embedding in document_embeddings]

    reshape_price_feature = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=-1))
    reshaped_price_features = [reshape_price_feature(inputs[fname]) for fname in inputs.keys() if '8-k' not in fname]
    time_series_input = tf.keras.layers.Concatenate()(reshaped_doc_embeddings + reshaped_price_features)

    time_series_lstm = tf.keras.layers.LSTM(lstm_layer_units)(time_series_input)

    output_layer = tf.keras.layers.Dense(units=1, kernel_initializer=output_kernel_init,
                                         bias_initializer=output_bias_init, name='adjusted_close_target_WFC')
    outputs = {'adjusted_close_target_WFC': output_layer(time_series_lstm)}


    model = tf.keras.Model(inputs, outputs, name='model_0')

    return model

# END Default Model Definitions

###########################
## END Model Definitions ##
###########################
