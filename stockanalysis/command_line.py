import pkgutil
import json
import click
from stockanalysis import pipelines
from stockanalysis.data import fetch_data

### TODO:
# 1) Complete pr
# 2) Complete TODO for pull_data then the rest


### TODO for pull_data:
# 1) for click option to be of type choice where the choices are valid SEC forms
# 2) make help readme better.
# 3) need to inform user about api keys and how this works with it etc...
@click.command()
@click.argument('path', type=click.Path(), nargs=1)
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

### TODO for run_pipeline:
# 1) Determine whether this tool implements enough customizability.
@click.command()
@click.argument('path', type=click.Path(exists=True, file_okay=False, writable=True), nargs=1)
@click.option('--custom', type=click.File('r'), nargs=1,
              help='Path to the json config file defining the parameters used in the implementation of the custom pipeline defined in the pipelines module.')
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
        custom_flag = True
        params = json.load(custom)
    else:
        click.echo('default pipeline was triggered')
        custom_flag = False
        default_config_file = pkgutil.get_data(__name__, 'default_config.json')
        params = json.loads(default_config_file)
    pipelines.run_pipelines(path, gpu_memory, custom_flag, **params)

@click.command()
# prompt for overwrite
@click.argument('path', type=click.Path(), nargs=1)
@click.option('--custom', type=click.File('r'), nargs=1,
              help='Path to the config file defining the custom parameters used in the implementation of the custom pipeline defined in the pipelines module.')
@click.option('--gpu_memory', type=int,
              help='GPU memory to allocate for training model')
@click.option('--ticker', '-t', 'tickers', type=str,
              default=['INTC'], help='bullshit', multiple=True)
@click.option('--form_type', '-f', 'form_types', type=str, default=['10-k'],
              help='SEC form type to download for the given stock tickers.',
              multiple=True)
def pr(path, custom, gpu_memory, tickers, form_types):
    fetch_data(path, tickers, form_types)
    if custom:
        click.echo('custom pipeline is triggered')
        params = json.load(custom)
        pipelines.custom_pipeline(path, gpu_memory, **params)
    else:
        click.echo('default pipeline is triggered')
        default_config_file = pkgutil.get_data(__name__, 'default_config.json')
        params = json.loads(default_config_file)
        pipelines.pipeline(path, gpu_memory, **params)




# Current implementation does not perform data checks on the data lying in paths
# to see if can be reused again, it will just overwrite




# current implementation of run_pipeline does not allow custom pipelines to
#  have custom parameters, custom pipelines must implement the same interface
#  as the default pipeline. This lowers customizablitiy, if this is an issue.
#  below might be a potential solution.

'''@click.group(invoke_without_command=True)
@click.argument('path', type=click.Path(sterisk form of **kwargs is used to pass a keyworded, variable-length argument dictionary to a function. Again, the two asterisks (**) are the important element here, as the word kwargs is conventionally used, though not enforced by the language.

Like *args, **kwargs can take however many arguments you would like to supply to it. However, **kwargs differs from *args in that you will need toexists=True), nargs=1)
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
