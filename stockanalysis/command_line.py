import pkgutil
import os
import json
import click
from stockanalysis import pipelines
from stockanalysis.data import fetch_data

@click.group()
def stockanalysis():
    """
    A package of tools for building statistical models of stock prices.
    """
    pass

@stockanalysis.command()
@click.option('--quandl', '-q', 'quandl', type=str,
              prompt='Enter your API key for Quandl\'s data services' ,
              help='API key for Quandl\'s services')
@click.option('--alpha', '-a', 'alphavantage', type=str,
              prompt='Enter your API key for Alphavantage\'s data services',
              help='API key for Alphavantage\'s services')
def config(quandl, alphavantage):
    """
    Configures API keys for stockanalysis tools.

    API keys required by this tool are Quandl's and/or Alphavantage's. Either
    Quandl's or Alphavantage's services are used in order to access data to
    train models on.
    """

    keys = {'alphavantage': alphavantage, 'quandl': quandl}

    api_keys_file = os.path.expanduser('~/.stockanalysis_keys.json')
    with open(api_keys_file, 'w') as f:
        json.dump(keys, f)

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
