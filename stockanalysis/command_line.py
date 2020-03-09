import pkgutil
import json
import click
from stockanalysis import pipelines
from stockanalysis.data import fetch_data

### TODO:
# 1) Complete TODO for pull_data then the rest

@click.group()
def stockanalysis():
    """
    dieiewowpodkd
    """
    pass

@stockanalysis.command()
@click.option('--quandl', '-q', 'quandl', help='API key for Quandl\'s services')
@click.option('--alpha', '-a', 'alphavantage', help='API key for Alphavantage\'s services')
def config(quandl, alphavantage):
    """
    Configures API keys for stockanalysis tools.
    """
    keys = {'alphavantage': alphavantage, 'quandl': quandl}
    with open('api_keys.json', 'w') as f:
        json.dump(keys, f)

### TODO for pull_data:
# 1) need to inform user about api keys and how this works with it etc...
# 2) make help readme better.
@stockanalysis.command()
@click.argument('path', type=click.Path(), nargs=1)
@click.argument('tickers', type=str, nargs=-1)
@click.option('--form_type', '-f', 'form_types',
              type=click.Choice(['8-k', '10-k', '10'], case_sensitive=True),
              help='SEC form type to download for the given stock tickers.',
              multiple=True)
def pull_data(path, tickers, form_types):
    """
    Pulls end of day stock data and optionally SEC forms for the
    given stock [TICKERS] and stores this data in the file pointed to by PATH.
    For more information about the  structure of how this data is stored in PATH,
    see: (github readme link)
    """

    fetch_data(path, tickers, form_types)

### TODO for run_pipeline:
# 1) Determine whether this tool implements enough customizability.
@stockanalysis.command()
@click.argument('path', type=click.Path(exists=True, file_okay=False, writable=True), nargs=1)
@click.option('--custom', type=click.File('r'), nargs=1,
              help='Path to the json config file defining the parameters used in the implementation of the custom pipeline defined in the pipelines module.')
@click.option('--gpu_memory', '-g', 'gpu_memory', type=int,
              help='GPU memory to allocate for training the model.')
def run_pipeline(path, custom, gpu_memory):
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
    pipelines.run_pipelines(path, gpu_memory, custom_flag, **params)

### TODO for pr:
# 1) Write better documentation and come up with a better name

@stockanalysis.command()
@click.argument('path', type=click.Path(), nargs=1)
@click.option('--custom', type=click.File('r'), nargs=1,
              help='Path to the config file defining the custom parameters used in the implementation of the custom pipeline defined in the pipelines module.')
@click.option('--gpu_memory', '-g', 'gpu_memory', type=int,
              help='GPU memory to allocate for training the model.')
@click.option('--ticker', '-t', 'tickers', type=str,
              default=['INTC'], help='Stock tickers to train model on.', multiple=True)
@click.option('--form_type', '-f', 'form_types',
              type=click.Choice(['8-k', '10-k', '10'], case_sensitive=True),
              default=['10-k'],
              help='SEC form type to download for the given stock tickers.',
              multiple=True)
def pr(path, custom, gpu_memory, tickers, form_types):
    """
    adsfadsfadsfads
    """
    fetch_data(path, tickers, form_types)
    if custom:
        click.echo('custom pipeline is triggered')
        params = json.load(custom)
        pipelines.custom_pipeline(path, gpu_memory, **params)
    else:
        click.echo('default pipeline is triggered')
        default_config_file = pkgutil.get_data(__name__, 'default_config.json')
        params = json.loads(default_config_file)
    pipelines.run_pipelines(path, gpu_memory, **params)
