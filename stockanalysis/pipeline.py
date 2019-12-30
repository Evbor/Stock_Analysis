
def pipeline(datastorage_path, stocks_to_model,
             doc_filename='documents', df_filename='raw.csv'):
    '''
    ML pipeline that trains a model, and saves it.
    '''

    df = pd.read_csv(os.path.join(path_to_data, df_filename), parse_dates=['timestamp'])
    preprocessed_df = preprocess_features(df, tickers, cut_off=20)

    return preprocessed_df
