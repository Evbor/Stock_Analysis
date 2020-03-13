import os
import pickle
import tensorflow as tf

#######################
## Model Definitions ##
#######################

# Default Model Definitions

def embedding_matrix(vocab, init):
    """
    Constructs the embedding matrix for specific init type for a pre initialized word embedding layer.

    :param vocab: dict, a mapping between keys of words, and values of unique integer identifiers for each word
    :param init: string, initialization type currently we only support glove initialization

    ---> numpy array of size (vocab length, embedding dimension) mapping each word encoding to a vector
    """

    if init == 'glove':
        glove_dir = 'research/model_v1/glove'

        try:
            with open(os.path.join(glove_dir, 'current_embedding.pickle'), 'rb') as f:
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
            with open(os.path.join(glove_dir, 'current_embedding.pickle'), 'wb') as f:
                pickle.dump(embedding_m, f)

    else:
        raise ValueError('init type not supported, init must be equal to "glove"')

    return embedding_m

def Word_Embedding(vocab, init,
                   embeddings_initializer='uniform', embeddings_regularizer=None,
                   activity_regularizer=None, embeddings_constraint=None,
                   mask_zero=False, input_length=None, **kwargs):

    """
    Creates a keras embedding layer specifically designed to embed the words specified in :param vocab:

    :param vocab: dict, representing the mapping between the words in corpus (keys) and their unique integer
                  encodings
    :param init: string or int, tells the layer how to initialize its embeddings. If of type int, then
                 it tells the layer to initialize its word embeddings with an embedding dimension of :param init:.
                 If of type string, then :param init: specifies the type of pretrained word embeddings we will be
                 initializing the embedding layer with

    ---> tf.keras.layers.Embedding
    """

    if isinstance(init, str):
        current_embedding_matrix = embedding_matrix(vocab, init)
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

def document_embedder_model(vocab, doc_embedding_size=100):
    input_doc = tf.keras.Input(shape=(None,), name='doc')
    word_embedding = Word_Embedding(vocab, init='glove', mask_zero=True, trainable=False)(input_doc)
    document_embedding = tf.keras.layers.LSTM(doc_embedding_size)(word_embedding)
    model = tf.keras.Model(input_doc, document_embedding, name='document_embedder')
    return model

def model_0(vocab, doc_embedding_size=100, lstm_layer_units=32,
            output_kernel_init=None, output_bias_init=None):

    if output_bias_init is not None:
        output_bias_init = tf.keras.initializers.Constant(output_bias_init)

    inputs = {'log_adj_daily_returns': tf.keras.Input(shape=(5,), name='log_adj_daily_returns', dtype=tf.float32),
              '8-k': tf.keras.Input(shape=(None,), name='8-k', dtype=tf.int64)}

    doc_embeddings = document_embedder_model(vocab, doc_embedding_size)(inputs['8-k'])

    reshape_doc_embeddings = tf.keras.layers.Lambda(lambda x: tf.keras.backend.stack([x for i in range(5)], axis=1))(doc_embeddings)
    reshape_price_features = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=-1))(inputs['log_adj_daily_returns'])
    time_series_input = tf.keras.layers.Concatenate()([reshape_doc_embeddings, reshape_price_features])

    time_series_lstm = tf.keras.layers.LSTM(lstm_layer_units)(time_series_input)

    output_layer = tf.keras.layers.Dense(units=1, kernel_initializer=output_kernel_init,
                                         bias_initializer=output_bias_init, name='log_adj_daily_returns_target')
    outputs = {'log_adj_daily_returns_target': output_layer(time_series_lstm)}


    model = tf.keras.Model(inputs, outputs, name='model_0')

    return model

# END Default Model Definitions

###########################
## END Model Definitions ##
###########################
