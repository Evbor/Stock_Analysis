import pkgutil
import os
import json
import click

from stockanalysis.data import fetch_data, download_file, unzip_file

@click.group()
def stockanalysis():
    """
    A package of tools for building statistical models of stock prices.
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
    Pulls data for training statistical models on.

    Pulls end of day stock data and optionally SEC forms, for the
    given stock [TICKERS] and stores this data in the file pointed to by PATH.
    For more information about the data storage structure see: (link)
    """

    fetch_data(path, tickers, source, form_types)

@stockanalysis.command()
@click.argument('path', type=click.Path(exists=True, file_okay=False, writable=True), nargs=1)
@click.argument('modelname', type=str, nargs=1)
@click.argument('modelversion', type=click.IntRange(min=0, max=None, clamp=False), nargs=1)
@click.option('--custom', type=click.File('r'), nargs=1,
              help='A json file defining parameter values for the custom pipeline\'s parameters')
@click.option('--gpu_memory', '-g', 'gpu_memory', type=int,
              help='GPU memory to allocate for training the model')
def run_pipeline(path, modelname, modelversion, custom, gpu_memory):
    """
    Runs ML pipelines to build statistical models.

    Runs a machine learning pipeline that trains a model on the dataset pointed
    to by [PATH]. The trained model is saved in a directory called models with
    [MODELNAME] and [MODELVERSION] number. If the custom option is not set
    then the default machine learning pipeline is run with its default
    configuration. Else the custom pipeline is run with its configuration defined
    in the file given as its argument. For more information on implementing
    custom pipelines see: (link)
    """

    from stockanalysis import pipelines

    if custom:
        click.echo('custom pipeline was triggered')
        custom_flag = True
        params = json.load(custom)
    else:
        click.echo('default pipeline was triggered')
        custom_flag = False
        default_config_file = pkgutil.get_data(__name__, 'default_config.json')
        params = json.loads(default_config_file)
    pipelines.run_pipelines(path, modelname, modelversion, gpu_memory, custom_flag, **params)

@stockanalysis.command()
@click.argument('path', type=click.Path(), nargs=1)
@click.argument('modelname', type=str, nargs=1)
@click.argument('modelversion', type=click.IntRange(min=0, max=None, clamp=False), nargs=1)
@click.option('--custom', type=click.File('r'), nargs=1,
              help='A json file defining parameter values for the custom pipeline\'s parameters')
@click.option('--gpu_memory', '-g', 'gpu_memory', type=int,
              help='GPU memory to allocate for training the model.')
@click.option('--ticker', '-t', 'tickers', type=str,
              default=['WFC', 'JPM', 'BAC', 'C'],
              help='Stock tickers to train model on.', multiple=True)
@click.option('--source', '-s', 'source',
              type=click.Choice(['quandl', 'alphavantage'], case_sensitive=True),
              default='alphavantage',
              help='Service used to source end of day stock ticker data')
@click.option('--form_type', '-f', 'form_types',
              type=click.Choice(['8-k', '10-k', '10'], case_sensitive=True),
              default=['8-k'],
              help='SEC form type to download for the given stock tickers',
              multiple=True)
def create_models(path, modelname, modelversion, custom, gpu_memory, tickers, source, form_types):
    """
    Creates models by pulling data and running a ML pipeline.

    Creates a statistical model by pulling the necessary data into a directory
    pointed to by [PATH]. Then running a machine learning pipeline which saves
    the trained model in a directory called models with [MODELNAME] and
    [MODELVERSION] number. If the custom option is not set then the default
    machine learning pipeline is run with its default configuration. Else the
    custom pipeline is run with its configuration defined in the file given as
    its argument. The dataset pulled is specified by providing the ticker,
    source, and form_type options. If no options are provided to specify the
    dataset, then the default dataset required by the default machine learning
    pipeline is pulled.
    """

    from stockanalysis import pipelines

    fetch_data(path, tickers, source, form_types)

    if custom:
        click.echo('custom pipeline was triggered')
        custom_flag = True
        params = json.load(custom)
    else:
        click.echo('default pipeline was triggered')
        custom_flag = False
        default_config_file = pkgutil.get_data(__name__, 'default_config.json')
        params = json.loads(default_config_file)
    pipelines.run_pipelines(path, modelname, modelversion, gpu_memory, custom_flag, **params)
