import click
from stockanalysis import pipelines
from stockanalysis.data import fetch_data


# TODOS for pull_data:
# 1) for click option to be of type choice where the choices are valid SEC forms
# 2) make help readme better.
# 3) need to inform user about api keys and how this works with it etc...
@click.command()
@click.argument('path', type=click.Path(exists=True), nargs=1)
@click.argument('tickers', type=str, nargs=-1)
@click.option('--form_type',
              '-f',
              'form_types',
              type=str,
              help='SEC form type to download for the given stock tickers.',
              multiple=True)
def pull_data(path, tickers, form_types):
    """
    Pulls end of day stock data and optionally SEC forms for the
    given stock [TICKERS] and stores this data in the file pointed to by PATH.
    For more information about the structure of how this data is stored in PATH,
    see: (github readme link)
    """
    fetch_data(path, tickers, form_types)


@click.command()
@click.option('--filename',
              type=click.Path(exists=True),
              prompt='Path to data file',
              help='Path to data file')
@click.option('--tickers',
              '-t',
              type=str,
              prompt='List of stock tickers to train model on must be a combination of: WFC, JPM, C, or BAC',
              help='List of stock tickers to train model on must be a combination of: WFC, JPM, C, or BAC',
              multiple=True)
@click.option('--gpu_memory',
              type=int,
              default=0,
              prompt='GPU memory to allocate for training set to 0 to not use GPU',
              help='GPU memory to allocate for training set to 0 to not use GPU')
def run_pipeline(filename, tickers, gpu_memory):
    if gpu_memory == 0:
        gpu_memory = None
    pipelines.pipeline(filename, tickers, gpu_memory=gpu_memory)

if __name__ == '__main__':
    pull_data()
