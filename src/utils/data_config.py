import timm.data
from timm.data.constants import *

def resolve_data_config(args, default_cfg={}, model=None, use_test_size=False, verbose=False):
    new_config = timm.data.resolve_data_config(args, default_cfg={}, model=None, use_test_size=False, verbose=False)

    # Resolve input/image size
    in_chans = 3
    if 'chans' in args and args['chans'] is not None:
        in_chans = args['chans']

    # resolve dataset source + model mean for normalization
    new_config['mean_source'] = IMAGENET_DEFAULT_MEAN
    if 'mean_source' in args and args['mean_source'] is not None:
        mean = tuple(args['mean_source'])
        if len(mean) == 1:
            mean = tuple(list(mean) * in_chans)
        else:
            assert len(mean) == in_chans
        new_config['mean_source'] = mean
    elif 'mean_source' in default_cfg:
        new_config['mean_source'] = default_cfg['mean_source']

    # resolve dataset source + model std deviation for normalization
    new_config['std_source'] = IMAGENET_DEFAULT_STD
    if 'std_source' in args and args['std_source'] is not None:
        std = tuple(args['std_source'])
        if len(std) == 1:
            std = tuple(list(std) * in_chans)
        else:
            assert len(std) == in_chans
        new_config['std_source'] = std
    elif 'std_source' in default_cfg:
        new_config['std_source'] = default_cfg['std_source']

    # resolve dataset target + model mean for normalization
    new_config['mean_target'] = IMAGENET_DEFAULT_MEAN
    if 'mean_target' in args and args['mean_target'] is not None:
        mean = tuple(args['mean_target'])
        if len(mean) == 1:
            mean = tuple(list(mean) * in_chans)
        else:
            assert len(mean) == in_chans
        new_config['mean_target'] = mean
    elif 'mean_target' in default_cfg:
        new_config['mean_target'] = default_cfg['mean_target']

    # resolve dataset target + model std deviation for normalization
    new_config['std_target'] = IMAGENET_DEFAULT_STD
    if 'std_target' in args and args['std_target'] is not None:
        std = tuple(args['std_target'])
        if len(std) == 1:
            std = tuple(list(std) * in_chans)
        else:
            assert len(std) == in_chans
        new_config['std_target'] = std
    elif 'std_target' in default_cfg:
        new_config['std_target'] = default_cfg['std_target']

    return new_config
