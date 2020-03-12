import pkgutil
import os
import json
import click
from stockanalysis import pipelines
from stockanalysis.data import fetch_data

### TODO for stockanalysis:
# 1) Write better documentation
@click.group()
def stockanalysis():
    """
    Package for performing stockanalysis on end of day stock data.
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

### TODO for pull_data:
# 1) make help readme better.
@stockanalysis.command()
@click.argument('path', type=click.Path(), nargs=1)
@click.argument('tickers', type=str, nargs=-1)
@click.option('--source', '-s', 'source',
              type=click.Choice(['quandl', 'alphavantage'], case_sensitive=True),
              default='alphavantage',
              help='Service used to source end of day stock ticker data from.')
@click.option('--form_type', '-f', 'form_types',
              type=click.Choice(['8-k', '10-k', '10'], case_sensitive=True),
              help='SEC form type to download for the given stock tickers.',
              multiple=True)
def pull_data(path, tickers, source, form_types):
    """
    Pulls end of day stock data and optionally SEC forms for the
    given stock [TICKERS] and stores this data in the file pointed to by PATH.
    For more information about the  structure of how this data is stored in PATH,
    see: (github readme link)
    """

    fetch_data(path, tickers, source, form_types)

### TODO for run_pipeline:
# 1) Write better documentation
# 2) Determine whether this tool implements enough customizability.
@stockanalysis.command()
@click.argument('path', type=click.Path(exists=True, file_okay=False, writable=True), nargs=1)
@click.argument('model_name', type=str, nargs=1)
@click.argument('model_version', type=click.IntRange(min=0, max=None, clamp=False), nargs=1)
@click.option('--custom', type=click.File('r'), nargs=1,
              help='Path to the json config file defining the parameters used in the implementation of the custom pipeline defined in the pipelines module.')
@click.option('--gpu_memory', '-g', 'gpu_memory', type=int,
              help='GPU memory to allocate for training the model.')
def run_pipeline(path, model_name, model_version, custom, gpu_memory):
    """
    Runs pipeline that trains a model on the data stored in [PATH]. If the
    custom option is not set, then the model trained is the default model
    defined in the pipelines module.
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
    pipelines.run_pipelines(path, model_name, model_version, gpu_memory, custom_flag, **params)

### TODO for pr:
# 1) Write better documentation and come up with a better name
@stockanalysis.command()
@click.argument('path', type=click.Path(), nargs=1)
@click.argument('model_name', type=str, nargs=1)
@click.argument('model_version', type=click.IntRange(min=0, max=None, clamp=False), nargs=1)
@click.option('--custom', type=click.File('r'), nargs=1,
              help='Path to the config file defining the custom parameters used in the implementation of the custom pipeline defined in the pipelines module.')
@click.option('--gpu_memory', '-g', 'gpu_memory', type=int,
              help='GPU memory to allocate for training the model.')
@click.option('--ticker', '-t', 'tickers', type=str,
              default=['INTC'], help='Stock tickers to train model on.', multiple=True)
@click.option('--source', '-s', 'source',
              type=click.Choice(['quandl', 'alphavantage'], case_sensitive=True),
              default='alphavantage',
              help='Service used to source end of day stock ticker data from.')
@click.option('--form_type', '-f', 'form_types',
              type=click.Choice(['8-k', '10-k', '10'], case_sensitive=True),
              default=['10-k'],
              help='SEC form type to download for the given stock tickers.',
              multiple=True)
def pr(path, model_name, model_version, custom, gpu_memory, tickers, source, form_types):
    """
    adsfadsfadsfads
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
    pipelines.run_pipelines(path, model_name, model_version, gpu_memory, custom_flag, **params)
