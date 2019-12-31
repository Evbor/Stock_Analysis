import click
from stockanalysis import pipelines



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
