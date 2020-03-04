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
    For more information about the  structure of how this data is stored in PATH,
    see: (github readme link)
    """
    fetch_data(path, tickers, form_types)

# TODOS for run_pipeline:
# 1) currently breaks in preprocessing, fix it.
# 2) Determine whether this tool implements enough customizability.

@click.command()
@click.argument('path', type=click.Path(exists=True), nargs=1)
@click.option('--custom', is_flag=True,
              help='If set, implements the custom pipeline defined in the pipelines module')
@click.option('--gpu_memory', type=int,
              help='GPU memory to allocate for training model')
def run_pipeline(path, custom, gpu_memory):
    """
    Runs pipeline that trains a model on the data stored in [PATH]. If the
    custom option is not set, then the model trained is the default model
    defined in the pipelines module.
    """
    if custom:
        click.echo('custom pipeline was triggered')
        pipelines.custom_pipeline(path, gpu_memory)
    else:
        click.echo('default pipeline was triggered')
        pipelines.pipeline(path, gpu_memory)




# current implementation of run_pipeline does not allow custom pipelines to
#  have custom parameters, custom pipelines must implement the same interface
#  as the default pipeline. This lowers customizablitiy, if this is an issue.
#  below might be a potential solution.

'''@click.group(invoke_without_command=True)
@click.argument('path', type=click.Path(exists=True), nargs=1)
@option('--gpu_memory', type=int, help='blahblahblah')
@click.pass_context
def run_pipe(ctx, path, gpu_memory):
    if gpu_memory == 0:
        gpu_memory = None
    ctx.ensure_object(dict)
    ctx.obj['PATH'] = path
    ctx.obj['GPU_MEMORY'] = gpu_memory
    if ctx.invoked_subcommand is None:
        pass # Run default pipeline with
        '''
