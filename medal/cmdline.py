"""
Tooling to initialize and run models from commandline
"""
import configargparse as ap
import torch

from . import model_configs as MC


def main(ns: ap.Namespace):
    """Initialize model and run from command-line"""

    # merge cmdline config with defaults
    config_overrides = ns.__dict__
    config = config_overrides.pop('modelconfig_class')(config_overrides)
    print('\n'.join(str((k, v)) for k, v in config.__dict__.items()
                    if not k.startswith('__')))

    # assign model to cuda device if necessary
    if config.device == 'cuda' and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        config.model = torch.nn.DataParallel(config.model)
    config.model.to(config.device)

    config.load_checkpoint()

    config.train()


def _add_subparser_find_configurable_attributes(kls):
    # get a list of configurable attributes from given class and parent classes
    keys = set()
    klass_seen = set()
    klass_lst = [kls]
    while klass_lst:
        klass = klass_lst.pop()
        for k in klass.__bases__:
            if k in klass_seen:
                continue
            klass_seen.add(k)
            klass_lst.append(k)
        keys.update({x for x in klass.__dict__ if not x.startswith('_')})
    return keys


def _add_subparser_arg(subparser, k, v, obj):
    """Add an argparse option via subparser.add_argument(...)
    for the given attribute k, with default value v, and where hasattr(obj, k)
    """
    g = subparser
    accepted_simple_types = (int, float, str)
    ku = k.replace('_', '-')
    if isinstance(v, bool):
        grp = g.add_mutually_exclusive_group()
        grp.add_argument('--%s' % ku, action='store_const', const=True)
        grp.add_argument(
            '--no-%s' % ku, action='store_const', const=False, dest=k)
    elif isinstance(v, accepted_simple_types):
        g.add_argument(
            '--%s' % ku, type=type(v), default=getattr(obj, k), help=' ')
    elif isinstance(v, (list, tuple)):
        if all(isinstance(x, accepted_simple_types) for x in v):
            g.add_argument(
                '--%s' % ku, nargs=len(v), default=v, help=' ',
                type=lambda inpt: type(types)(
                    [typ(x) for typ, x in zip(types, inpt)]))
        else:
            g.add_argument('--%s' % ku, nargs=len(v), type=v[0])
    elif any(v is x for x in accepted_simple_types):
        g.add_argument('--%s' % ku, type=v)


def add_subparser(subparsers, name: str, modelconfig_class: type):
    """
    Automatically add parser options for attributes in given class and
    all of its parent classes
    """
    g = subparsers.add_parser(
        #  name, formatter_class=ap.RawDescriptionHelpFormatter)
        name, formatter_class=ap.ArgumentDefaultsHelpFormatter)
    g.add_argument(
        '--modelconfig_class', help=ap.SUPPRESS, default=modelconfig_class)

    # add an argument for each configurable key that we can work with
    keys = _add_subparser_find_configurable_attributes(modelconfig_class)
    for k in keys:
        v = getattr(modelconfig_class, k)
        _add_subparser_arg(g, k, v, modelconfig_class)


def build_arg_parser():
    """Returns a parser to handle command-line arguments"""
    p = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter)
    sp = p.add_subparsers(help='The model configuration to work with')
    sp.required = True
    sp.dest = 'model_configuration_name'

    for kls_name in dir(MC):
        if kls_name.startswith("_"):
            continue
        # add all available model config classes as command line options
        mc_obj = getattr(MC, kls_name)
        if isinstance(mc_obj, type) \
                and issubclass(mc_obj, MC.feedforward.FeedForwardModelConfig):
            add_subparser(sp, mc_obj.__name__, mc_obj)
    return p
