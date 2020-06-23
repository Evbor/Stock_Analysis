import pkgutil
import click
import os
import json
import pickle

@click.group()
def stockanalysis():
    """
    A package of tools for building machine learning models of stock prices.
    """
    pass

@stockanalysis.command()
@click.option('--quandl', '-q', 'quandl', type=str,
              help='API key for Quandl\'s services')
@click.option('--alpha', '-a', 'alphavantage', type=str,
              help='API key for Alphavantage\'s services')
@click.option('--glove', '-g', 'glove', type=click.Path(file_okay=False, writable=True),
              help='path to directory storing pre-trained GloVe vectors file. If the value passed is "default" then the default path is used.')
def config(quandl, alphavantage, glove):
    """
    Configures stockanalysis tools.

    Can be used to configure the API keys for data scraping, as well as the
    directory storing the pre-trained GloVe vectors file: glove840B.300d.txt.
    For more information on GloVe see: nlp.stanford.edu/projects/glove/.
    API keys required by this tool are Quandl's and/or Alphavantage's. Either
    Quandl's or Alphavantage's services are used in order to access data to
    train models on.
    """

    from stockanalysis.data import download_file, unzip_file

    # Loading config file if it exists
    config_dir = os.path.join('~', '.stockanalysis')

    try:
        with open(os.path.expanduser(os.path.join(config_dir, 'config.json')), 'r') as f:
            configuration = json.load(f)
    except FileNotFoundError:
        if not os.path.isdir(os.path.expanduser(config_dir)):
            os.makedirs(config_dir)
        configuration = {}

    api_keys = configuration.get('API_Keys', {})
    model_resources = configuration.get('Model_Resources', {})

    # Modifying keys
    if quandl:
        api_keys['quandl'] = quandl
    if alphavantage:
        api_keys['alphavantage'] = alphavantage

    # Modifying GloVe path if it exists else
    if not glove:
        glove = model_resources.get('glove',
                                    os.path.join(config_dir,
                                                 'model_resources', 'glove'))
    elif glove == 'default':
        glove = os.path.join(config_dir, 'model_resources', 'glove')

    model_resources['glove'] = glove

    # Writing Configuration File
    new_configuration = {
                         'API_Keys': api_keys,
                         'Model_Resources': model_resources
                         }

    with open(os.path.expanduser(os.path.join(config_dir, 'config.json')), 'w') as f:
        json.dump(new_configuration, f)

    # Testing if GloVe files exist and downloading it if it doesn't
    if not os.path.isfile(os.path.expanduser(os.path.join(glove, 'glove.840B.300d.txt'))):
        if click.confirm('Pre-trained GloVe vectors file not found. This file is required by some of the modeling process and is about 3 GB in size. Do you want to download the file now?'):
            if not os.path.isdir(os.path.expanduser(glove)):
                os.makedirs(glove)
            click.echo('Installing {} to location {}'.format('glove840B.300d.txt', glove))
            url = 'http://nlp.stanford.edu/data/'
            zipfile = 'glove.840B.300d.zip'
            click.echo('Downloading {}'.format(zipfile))
            zipfile_path = download_file(url + zipfile, os.path.join(glove, zipfile))
            click.echo('Unzipping file')
            unzip_file(zipfile_path)
            os.remove(zipfile_path)

@stockanalysis.command()
@click.argument('path', type=click.Path(), nargs=1)
@click.argument('tickers', type=str, nargs=-1)
@click.option('--source', '-s', 'source',
              type=click.Choice(['quandl', 'alphavantage'], case_sensitive=True),
              default='alphavantage',
              help='Service used to source end of day stock ticker data')
@click.option('--form_type', '-f', 'form_types',
              type=click.Choice(['8-k', '10-k', '10'], case_sensitive=True),
              help='SEC form type to download for the given stock tickers',
              multiple=True)
def pull_data(path, tickers, source, form_types):
    """
    Pulls data for training models on.

    Pulls end of day stock data and optionally SEC forms, for the
    given stock [TICKERS] and stores this data in the file pointed to by PATH.
    For more information about the data storage structure see: (link)
    """

    from stockanalysis.data import fetch_data

    fetch_data(path, tickers, source, form_types)

@stockanalysis.command()
@click.argument('path_to_data', type=click.Path(), nargs=1)
@click.argument('path_to_metadata', type=click.Path(), nargs=1)
@click.argument('path_to_models', type=click.Path(), nargs=1)
@click.option('--gpu_memory', '-g', 'gpu_memory', type=int, default=0,
              help='GPU memory to allocate for training the model')
def pipeline(path_to_data, path_to_metadata, path_to_models, gpu_memory):
    """
    Runs ML pipeline to build machine learning models.

    Runs a machine learning pipeline that trains a model. The data consumed by
    models will be stored in the directory pointed to by [PATH_TO_DATA]. While
    meta-data produced by the pipeline will be stored in the directory pointed
    to by [PATH_TO_METADATA]. Models produced by the pipeline will be stored in
    the directory pointed to by [PATH_TO_MODELS]. The model deployed by the
    pipeline exists at path: [PATH_TO_MODELS]/deployed_model. For more
    information regarding the current deployed pipeline see: (link)
    """

    import stockanalysis.pipelines as p

    default_config_file = pkgutil.get_data(__name__, 'default_config.pickle')
    config = pickle.loads(default_config_file)
    p.pipeline(path_to_metadata, path_to_data, path_to_models, gpu_memory, config)
