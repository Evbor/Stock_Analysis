import click
from stockanalysis import pipelines
from stockanalysis.data import fetch_data

## complete command line application of fetchdata look into click arguements 

@click.command()
@click.option('--endpoint',
              type=click.Path(exists=True),
              prompt='Path to file to store data in.',
              help='Path to file to store data in.')
@click.option('--tickers',
              type=str,
              prompt='Stock tickers to pull data for.',
              help='Stock tickers to pull data for.',
              multiple=True)
@click.option('--form_types',
              type=str,
              prompt='SEC form types to pull for the given stock tickers.',
              help='SEC form types to pull for the given stock tickers.',
              multiple=True)
def pull_data(endpoint, tickers, form_types):
    fetch_data(endpoint, tickers, form_types)


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
