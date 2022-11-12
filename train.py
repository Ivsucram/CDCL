#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
from cmath import isnan
import logging
import os
import time

from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, \
    LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_sync_batchnorm, model_parameters, set_fast_norm
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

try:
    from apex import amp
    from torch.nn.parallel import DistributedDataParallel as ApexDDP
    from torch.nn.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False


from src.utils.data_config import resolve_data_config
from src.utils.dataset_factory import create_dataset
from src.utils.loader_factory import create_loaders
from src.utils.center_aware_pseudo import CenterAwarePseudoModule
from src.utils.distil_loss import DistilLoss
from src.utils.memory_manager import RehearsalMemoryManager


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='CLCD with timm')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside of the dataset group because it is positional.
parser.add_argument('--data_dir', metavar='DIR',
                    help='path to dataset')
group.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
group.add_argument('--dataset_source', '-ds', metavar='NAME', default=None,
                    help='dataset source (CLCD)')
group.add_argument('--dataset_target', '-dt', metavar='NAME', default=None,
                    help='dataset source (CLCD)')
group.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
group.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
group.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
group.add_argument('--mean_source', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset source')
group.add_argument('--std_source', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset source')
group.add_argument('--mean_target', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset target')
group.add_argument('--std_target', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset target')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='torch.jit.script the full model')
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                    help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
group.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                    help='Enable gradient checkpointing through model blocks/stages')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                    help='layer-wise learning rate decay (default: None)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
group.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
group.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                    help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
group.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
group.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
group.add_argument('--aug-repeats', type=float, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
group.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
group.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
group.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
group.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
group.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
group.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
group.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
group.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
group.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
group.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
group.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
group.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
group.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
group.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
group.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
group.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
group.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
group.add_argument('--eval-metric', default='cil_top1_t', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "cil_top1_t"')
group.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument("--local-rank", default=0, type=int)
group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
group.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
group.add_argument('--device-id', type=int, default=0, metavar='N', 
                    help='If you have multiples GPU, select which single GPU will run this code (default: 0)')

group.add_argument('--alpha-1', type=float, default=1., metavar='N',
                   help='Loss_s coefficient (default 1.0)')
group.add_argument('--alpha-2', type=float, default=1., metavar='N',
                   help='Loss_t coefficient (default 1.0)')
group.add_argument('--alpha-3', type=float, default=1., metavar='N',
                   help='Loss_d coefficient (default 1.0)')
group.add_argument('--alpha-4', type=float, default=0., metavar='N',
                   help='Loss_s coefficient (default 0.0)')
group.add_argument('--alpha-5', type=float, default=1., metavar='N',
                   help='Loss_i coefficient (default 1.0)')
group.add_argument('--alpha-6', type=float, default=1., metavar='N',
                   help='Loss_r coefficient (default 1.0)')


def _parse_args(config_path=None):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config or config_path is not None:
        with open(args_config.config if config_path is None else config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()
    # args, args_text = _parse_args(config_path='configs/datasets/usps_mnist.yml') # Used during development

    args.prefetcher = False
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = f'cuda:{args.device_id}'
    torch.cuda.set_device(args.device)
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        if 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.getenv('LOCAL_RANK'))
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    if args.rank == 0 and args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        args=args)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)
    if args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')


    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets
    dataset_source_train = create_dataset(args.dataset_source, is_training=True, args=args)
    dataset_source_eval = create_dataset(args.dataset_source, is_training=False, args=args)

    dataset_target_train = create_dataset(args.dataset_target, is_training=True, args=args)
    dataset_target_eval = create_dataset(args.dataset_target, is_training=False, args=args)
    if args.dataset_source == 'visda':
        dataset_source_train = create_dataset(args.dataset_source, is_training=True, args=args)
        dataset_source_eval = create_dataset(args.dataset_source, is_training=True, args=args)
    if args.dataset_target == 'visda':
        dataset_target_train = create_dataset(args.dataset_target, is_training=False, args=args)
        dataset_target_eval = create_dataset(args.dataset_target, is_training=False, args=args)

    # create data loaders w/ augmentation pipeiine
    (loader_source_train, loader_source_eval,
     loader_target_train, loader_target_eval,
     num_aug_splits, mixup_active, mixup_fn) = create_loaders(
        dataset_source_train, dataset_source_eval,
        dataset_target_train, dataset_target_eval,
        data_config, args
    )

    memory_manager = RehearsalMemoryManager(args=args)

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    distil_loss_fn = DistilLoss().cuda()
    loss_kl_fn = nn.KLDivLoss(reduction="batchmean").cuda()
    # loss_kl_fn = nn.MSELoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    cil1_R_matrix = torch.zeros((args.tasks, args.tasks))
    til1_R_matrix = torch.zeros((args.tasks, args.tasks))
    cil1_b_matrix = torch.zeros(args.tasks)
    til1_b_matrix = torch.zeros(args.tasks)

    cil5_R_matrix = torch.zeros((args.tasks, args.tasks))
    til5_R_matrix = torch.zeros((args.tasks, args.tasks))
    cil5_b_matrix = torch.zeros(args.tasks)
    til5_b_matrix = torch.zeros(args.tasks)

    try:
        for task in range(args.tasks):
            (loader_source_train, _, loader_target_train, _, num_aug_splits, mixup_active, mixup_fn) = create_loaders(
                dataset_source_train, dataset_source_eval, dataset_target_train, dataset_target_eval, data_config, args, task
            )

            for epoch in range(start_epoch, num_epochs):
                if args.distributed and hasattr(loader_source_train.sampler, 'set_epoch'):
                    loader_source_train.sampler.set_epoch(epoch)

                if epoch == 0:
                    (_, loader_source_eval, _, loader_target_eval, num_aug_splits, mixup_active, mixup_fn) = create_loaders(
                        dataset_source_train, dataset_source_eval, dataset_target_train, dataset_target_eval, data_config, args, task
                    )

                    eval_metrics = validate(model, (loader_source_eval, loader_target_eval), validate_loss_fn, args, amp_autocast=amp_autocast, til_task=task, cil_task=task)
                    cil1_b_matrix[task] = eval_metrics['cil_top1_t']
                    til1_b_matrix[task] = eval_metrics['til_top1_t']
                    cil5_b_matrix[task] = eval_metrics['cil_top5_t']
                    til5_b_matrix[task] = eval_metrics['til_top5_t']

                # if epoch == args.epochs:
                #     memory_manager.increment_task()

                train_metrics = train_one_epoch(
                    epoch, model, (loader_source_train, loader_target_train), optimizer, (train_loss_fn, distil_loss_fn, loss_kl_fn), args,
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn, task=task,
                    memory_manager=memory_manager, last_epoch = (epoch == num_epochs - 1))

                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    if args.local_rank == 0:
                        _logger.info("Distributing BatchNorm running means and vars")
                    utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

                for test_task in range(task+1):

                    (_, loader_source_eval, _, loader_target_eval, num_aug_splits, mixup_active, mixup_fn) = create_loaders(
                        dataset_source_train, dataset_source_eval, dataset_target_train, dataset_target_eval, data_config, args, test_task
                    )

                    eval_metrics = validate(model, (loader_source_eval, loader_target_eval), validate_loss_fn, args, amp_autocast=amp_autocast, til_task=test_task, cil_task=task)

                    if model_ema is not None and not args.model_ema_force_cpu:
                        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                            utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                        ema_eval_metrics = validate(
                            model_ema.module, (loader_source_eval, loader_target_eval), validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)', til_task=test_task, cil_task=task)
                        eval_metrics = ema_eval_metrics

                    if lr_scheduler is not None:
                        # step LR for next epoch
                        lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

                    if output_dir is not None:
                        utils.update_summary(
                            epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                            write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb, train_task=task, test_task=test_task)

                    if epoch == num_epochs - 1:
                        cil1_R_matrix[test_task][task] = eval_metrics['cil_top1_t']
                        til1_R_matrix[test_task][task] = eval_metrics['til_top1_t']
                        cil5_R_matrix[test_task][task] = eval_metrics['cil_top5_t']
                        til5_R_matrix[test_task][task] = eval_metrics['til_top5_t']

                    # if saver is not None:
                        # save proper checkpoint with eval metric
                        # save_metric = eval_metrics[eval_metric]
                        # best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass

    cil_acc1, cil_fwt1, cil_bwt1, cil_fgt1 = cil1_R_matrix.T[-1].mean().item(), (cil1_R_matrix.T[-1][1:] - cil1_b_matrix[1:]).mean().item(), (cil1_R_matrix.T[-1][0:-1] - cil1_R_matrix.diagonal()[0:-1]).mean().item(), ((cil1_R_matrix[0:-1].T - cil1_R_matrix.T[-1][0:-1]).max(0)[0]).mean().item()
    cil_acc5, cil_fwt5, cil_bwt5, cil_fgt5 = cil5_R_matrix.T[-1].mean().item(), (cil5_R_matrix.T[-1][1:] - cil5_b_matrix[1:]).mean().item(), (cil5_R_matrix.T[-1][0:-1] - cil5_R_matrix.diagonal()[0:-1]).mean().item(), ((cil5_R_matrix[0:-1].T - cil5_R_matrix.T[-1][0:-1]).max(0)[0]).mean().item()
    til_acc1, til_fwt1, til_bwt1, til_fgt1 = til1_R_matrix.T[-1].mean().item(), (til1_R_matrix.T[-1][1:] - til1_b_matrix[1:]).mean().item(), (til1_R_matrix.T[-1][0:-1] - til1_R_matrix.diagonal()[0:-1]).mean().item(), ((til1_R_matrix[0:-1].T - til1_R_matrix.T[-1][0:-1]).max(0)[0]).mean().item()
    til_acc5, til_fwt5, til_bwt5, til_fgt5 = til5_R_matrix.T[-1].mean().item(), (til5_R_matrix.T[-1][1:] - til5_b_matrix[1:]).mean().item(), (til5_R_matrix.T[-1][0:-1] - til5_R_matrix.diagonal()[0:-1]).mean().item(), ((til5_R_matrix[0:-1].T - til5_R_matrix.T[-1][0:-1]).max(0)[0]).mean().item()

    print(f'{cil_acc1=:5.2f} {cil_fwt1=:5.2f} {cil_bwt1=:5.2f} {cil_fgt1=:5.2f}')
    print(f'{cil_acc5=:5.2f} {cil_fwt5=:5.2f} {cil_bwt5=:5.2f} {cil_fgt5=:5.2f}')
    print(f'{til_acc1=:5.2f} {til_fwt1=:5.2f} {til_bwt1=:5.2f} {til_fgt1=:5.2f}')
    print(f'{til_acc5=:5.2f} {til_fwt5=:5.2f} {til_bwt5=:5.2f} {til_fgt5=:5.2f}')

    if args.log_wandb:
        rowd = OrderedDict()
        rowd.update([('CIL ACC @ 1', cil_acc1)])
        rowd.update([('CIL FWT @ 1', cil_fwt1)])
        rowd.update([('CIL BWT @ 1', cil_bwt1)])
        rowd.update([('CIL FGT @ 1', cil_fgt1)])

        rowd.update([('CIL ACC @ 5', cil_acc5)])
        rowd.update([('CIL FWT @ 5', cil_fwt5)])
        rowd.update([('CIL BWT @ 5', cil_bwt5)])
        rowd.update([('CIL FGT @ 5', cil_fgt5)])

        rowd.update([('TIL ACC @ 1', til_acc1)])
        rowd.update([('TIL FWT @ 1', til_fwt1)])
        rowd.update([('TIL BWT @ 1', til_bwt1)])
        rowd.update([('TIL FGT @ 1', til_fgt1)])

        rowd.update([('TIL ACC @ 5', til_acc5)])
        rowd.update([('TIL FWT @ 5', til_fwt5)])
        rowd.update([('TIL BWT @ 5', til_bwt5)])
        rowd.update([('TIL FGT @ 5', til_fgt5)])
        wandb.log(rowd)

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None, task=0, memory_manager=None, last_epoch=False):

    loader_target = loader[1]
    loader = loader[0]
    loss_distil_fn = loss_fn[1]
    loss_kl_fn = loss_fn[2]
    loss_cross_entropy_fn = loss_fn[0]

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = utils.AverageMeter()
    sample_pairs_m = utils.AverageMeter()
    losses_accumulator_s_m = utils.AverageMeter()
    losses_accumulator_t_m = utils.AverageMeter()
    losses_accumulator_d_m = utils.AverageMeter()
    losses_accumulator_i_m = utils.AverageMeter()
    losses_accumulator_a_m = utils.AverageMeter()
    losses_accumulator_r_m = utils.AverageMeter()

    mask = torch.zeros(args.num_classes).cuda()
    for i in range(mask.size(0)):
        if i < (args.num_classes // args.tasks * (task + 1)):            
            mask[i] = 1

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    dataloader_target_iterator = iter(loader_target)
    
    loader_memory = memory_manager.dataset_loader(task=task-1, mix_tasks=True) if task > 0 else None
    if loader_memory is not None:
        dataloader_memory_iterator = iter(loader_memory)

    for batch_idx, (input_source, label_source, _) in enumerate(loader):
        last_batch = batch_idx == last_idx
        if not args.prefetcher:
            input_source, label_source = input_source.cuda(), label_source.cuda()
            if mixup_fn is not None:
                input_source, label_source = mixup_fn(input_source, label_source)
        if args.channels_last:
            input_source = input_source.contiguous(memory_format=torch.channels_last)

        loss_s, loss_t, loss_d = torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda() # source, target, distil
        loss_a, loss_i, loss_r = torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda() # accumulator, injection, replay
        if epoch >= args.warmup_epochs:
            if batch_idx == 0:
                _logger.info('Computing pseudo-labels centroids')
                center_aware_pseudo_module = CenterAwarePseudoModule(model, loader_target, task=task, args=args)
                model.train()
                # if last_epoch:
                if epoch >= args.epochs:
                    _logger.info('Slower epoch: Saving confident paired samples into memory')

            with amp_autocast():
                try:
                    (input_target, _, _) = next(dataloader_target_iterator)
                except StopIteration:
                    dataloader_target_iterator = iter(loader_target)
                    (input_target, _, _) = next(dataloader_target_iterator)
                if not args.prefetcher:
                    input_target = input_target.cuda()
                if args.channels_last:
                    input_target = input_source.contiguous(memory_format=torch.channels_last)

                (inj_input_source, inj_input_target, inj_label_source), (acc_input_source, acc_input_target, acc_label_source) = center_aware_pseudo_module.reorder_datasets2(model, input_source, label_source, input_target, task=task)
                model.train()
                
                if inj_input_source is None:
                    sample_pairs_m.update(0)
                    injection_output, accumulator_output, _, previous_k_w, (k_w, k_b) = model(input_source, task=task)
                    accumulator_output = accumulator_output * torch.transpose(mask, 0, 0)
                    loss_i += loss_cross_entropy_fn(injection_output, label_source - (args.num_classes // args.tasks * task))
                    loss_s += loss_cross_entropy_fn(accumulator_output, label_source)
                else:    
                    sample_pairs_m.update(input_source.size(0))

                    ((injection_output, injection_output_target, injection_output_fusion),
                    (accumulator_output, accumulator_output_target, accumulator_output_fusion),
                    _,
                    (previous_k_w, previous_k_b),
                    (k_w, k_b)) = model(inj_input_source, inj_input_target, task=task)

                    accumulator_output, accumulator_output_target, accumulator_output_fusion = accumulator_output * torch.transpose(mask, 0, 0), accumulator_output_target * torch.transpose(mask, 0, 0), accumulator_output_fusion * torch.transpose(mask, 0, 0)

                    loss_s += loss_cross_entropy_fn(accumulator_output, inj_label_source)
                    loss_t += loss_cross_entropy_fn(accumulator_output_target, inj_label_source)
                    loss_d += loss_distil_fn(accumulator_output_target, accumulator_output_fusion)

                    loss_i += loss_cross_entropy_fn(injection_output, inj_label_source - (args.num_classes // args.tasks * task))
                    loss_i += loss_cross_entropy_fn(injection_output_target, inj_label_source - (args.num_classes // args.tasks * task))
                    loss_i += loss_distil_fn(injection_output_target, injection_output_fusion)

                # if last_epoch and inj_input_source is not None:
                if epoch >= args.epochs and inj_input_source is not None:
                    if batch_idx == 0:
                        memory_manager.zero_task(task)
                    for xs, xt, ys, ils, ilt, als, alt in zip(inj_input_source, inj_input_target, inj_label_source, injection_output, injection_output_target, accumulator_output, accumulator_output_target):
                        memory_manager.add_sample(xs, xt, ys, ils, ilt, als, alt, task)
        else:
            with amp_autocast():
                injection_output, accumulator_output, _, (previous_k_w, previous_k_b), (k_w, k_b) = model(input_source, task=task)
                accumulator_output = accumulator_output * torch.transpose(mask, 0, 0)
                loss_i += loss_cross_entropy_fn(injection_output, label_source - (args.num_classes // args.tasks * task))
                loss_s += loss_cross_entropy_fn(accumulator_output, label_source)

        if epoch >= args.epochs:
            loader_memory = memory_manager.dataset_loader(task=task, mix_tasks=True)
            dataloader_memory_iterator = iter(loader_memory)

        if task > 0 or epoch >= args.epochs:
            with amp_autocast():
                if task > 0:
                    loss_a += torch.norm(torch.gradient(previous_k_w, dim=0)[0].mean(0).unsqueeze(0) * loss_i * (k_w - previous_k_w), p=1)
                    loss_a += torch.norm(torch.gradient(previous_k_b, dim=0)[0].mean(0).unsqueeze(0) * loss_i * (k_b - previous_k_b), p=1)

                try:
                    (source_memory, target_memory, label_memory, _, _, acc_source_logit, acc_target_logit) = next(dataloader_memory_iterator)
                except StopIteration:
                    dataloader_memory_iterator = iter(loader_memory)
                    (source_memory, target_memory, label_memory, _, _, acc_source_logit, acc_target_logit) = next(dataloader_memory_iterator)

                source_memory, target_memory, label_memory = source_memory.cuda(), target_memory.cuda(), label_memory.cuda()
                acc_source_logit, acc_target_logit = acc_source_logit.cuda(), acc_target_logit.cuda()

                (_, (acc_memory_output, acc_memory_output_target, acc_memory_output_fusion), _, _, _) = model(source_memory, target_memory, task=task)

                acc_memory_output, acc_memory_output_target, acc_memory_output_fusion = acc_memory_output * torch.transpose(mask, 0, 0), acc_memory_output_target * torch.transpose(mask, 0, 0), acc_memory_output_fusion * torch.transpose(mask, 0, 0)

                loss_r += loss_kl_fn(F.softmax(acc_memory_output, dim=-1), F.softmax(acc_source_logit, dim=-1))
                loss_r += loss_kl_fn(F.softmax(acc_memory_output_target, dim=-1), F.softmax(acc_target_logit, dim=-1))
                loss_r += loss_cross_entropy_fn(acc_memory_output, label_memory)
                loss_r += loss_cross_entropy_fn(acc_memory_output_target, label_memory)
                loss_r += loss_distil_fn(acc_memory_output_target, acc_memory_output_fusion)

        alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6 = args.alpha_1, args.alpha_2, args.alpha_3, args.alpha_4, args.alpha_5, args.alpha_6
        loss = alpha_1 * loss_s + alpha_2 * loss_t + alpha_3 * loss_d + alpha_4 * loss_a + alpha_5 * loss_i + alpha_6 * loss_r

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                utils.dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if not args.distributed:
            losses_accumulator_s_m.update(loss_s.item(), input_source.size(0))
            losses_accumulator_t_m.update(loss_t.item(), input_source.size(0))
            losses_accumulator_d_m.update(loss_d.item(), input_source.size(0))
            losses_accumulator_i_m.update(loss_i.item(), input_source.size(0))
            losses_accumulator_a_m.update(loss_a.item(), input_source.size(0))
            losses_accumulator_r_m.update(loss_r.item(), input_source.size(0))

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_accumulator_s_m.update(reduced_loss.item(), input_source.size(0))

            if True: #args.local_rank == 0:
                if epoch >= args.warmup_epochs:
                    _logger.info(
                        'Task {0} '
                        'Train: {1} [{2:>4d}/{3} ({4:>3.0f}%)]  '
                        'Loss_s: {loss_s.val:#.4g} ({loss_s.avg:#.3g})  '
                        'Loss_t: {loss_t.val:#.4g} ({loss_t.avg:#.3g})  '
                        'Loss_d: {loss_d.val:#.4g} ({loss_d.avg:#.3g})  '
                        'Loss_i: {loss_i.val:#.4g} ({loss_i.avg:#.3g})  '
                        'Loss_a: {loss_a.val:#.4g} ({loss_a.avg:#.3g})  '
                        'Loss_r: {loss_r.val:#.4g} ({loss_r.avg:#.3g})  '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}   '
                        'Pairs: {pairs.val}/{pairs.sum} ({pairs.avg:#.3g})'.format(
                            task, epoch,
                            batch_idx, len(loader),
                            100. * batch_idx / last_idx,
                            loss_s=losses_accumulator_s_m,
                            loss_t=losses_accumulator_t_m,
                            loss_d=losses_accumulator_d_m,
                            loss_i=losses_accumulator_i_m,
                            loss_a=losses_accumulator_a_m,
                            loss_r=losses_accumulator_r_m,
                            batch_time=batch_time_m,
                            rate=input_source.size(0) * args.world_size / batch_time_m.val,
                            rate_avg=input_source.size(0) * args.world_size / batch_time_m.avg,
                            lr=lr, pairs=sample_pairs_m))
                else:
                    _logger.info(
                        'Task {0} '
                        'Train: {1} [{2:>4d}/{3} ({4:>3.0f}%)]  '
                        'Loss_s: {loss_s.val:#.4g} ({loss_s.avg:#.3g})  '
                        'Loss_i: {loss_i.val:#.4g} ({loss_i.avg:#.3g})  '
                        'Loss_a: {loss_a.val:#.4g} ({loss_a.avg:#.3g})  '
                        'Loss_r: {loss_r.val:#.4g} ({loss_r.avg:#.3g})  '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}'.format(
                            task, epoch,
                            batch_idx, len(loader),
                            100. * batch_idx / last_idx,
                            loss_s=losses_accumulator_s_m,
                            loss_i=losses_accumulator_i_m,
                            loss_a=losses_accumulator_a_m,
                            loss_r=losses_accumulator_r_m,
                            batch_time=batch_time_m,
                            rate=input_source.size(0) * args.world_size / batch_time_m.val,
                            rate_avg=input_source.size(0) * args.world_size / batch_time_m.avg,
                            lr=lr))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input_source,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_accumulator_s_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_accumulator_s_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix='', til_task=0, cil_task=0):
    batch_time_m = utils.AverageMeter()
    cil_losses_s_m = utils.AverageMeter()
    cil_losses_t_m = utils.AverageMeter()
    til_losses_s_m = utils.AverageMeter()
    til_losses_t_m = utils.AverageMeter()
    cil_top1_s_m = utils.AverageMeter()
    cil_top5_s_m = utils.AverageMeter()
    cil_top1_t_m = utils.AverageMeter()
    cil_top5_t_m = utils.AverageMeter()
    til_top1_s_m = utils.AverageMeter()
    til_top5_s_m = utils.AverageMeter()
    til_top1_t_m = utils.AverageMeter()
    til_top5_t_m = utils.AverageMeter()

    loader_target = loader[1]
    loader = loader[0]

    model.eval()

    end = time.time()
    last_idx = len(loader_target) - 1
    with torch.no_grad():
        for batch_idx, (input, label, _) in enumerate(loader_target):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input, label = input.cuda(), label.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                _, cil_output, _, _, _ = model(input, task=cil_task)
                til_output, _, _, _, _ = model(input, task=til_task)
            if isinstance(cil_output, (tuple, list)): cil_output = cil_output[0]
            if isinstance(til_output, (tuple, list)): til_output = til_output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                cil_output = cil_output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                til_output = til_output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                label = label[0:label.size(0):reduce_factor]

            loss_cil = loss_fn(cil_output, label)
            loss_til = loss_fn(til_output, label - (args.num_classes // args.tasks * til_task))
            cil_acc1, cil_acc5 = utils.accuracy(cil_output, label, topk=(1, 5))
            til_acc1, til_acc5 = utils.accuracy(til_output, label - (args.num_classes // args.tasks * til_task), topk=(1, 5))

            if args.distributed:
                reduced_loss_cil = utils.reduce_tensor(loss_cil.data, args.world_size)
                cil_acc1 = utils.reduce_tensor(cil_acc1, args.world_size)
                cil_acc5 = utils.reduce_tensor(cil_acc5, args.world_size)
            else:
                reduced_loss_cil = loss_cil.data

            cil_losses_t_m.update(reduced_loss_cil.item(), input.size(0))
            cil_top1_t_m.update(cil_acc1.item(), cil_output.size(0))
            cil_top5_t_m.update(cil_acc5.item(), cil_output.size(0))

            if args.distributed:
                reduced_loss_til = utils.reduce_tensor(loss_til.data, args.world_size)
                til_acc1 = utils.reduce_tensor(til_acc1, args.world_size)
                til_acc5 = utils.reduce_tensor(til_acc5, args.world_size)
            else:
                reduced_loss_til = loss_til.data

            torch.cuda.synchronize()

            til_losses_t_m.update(reduced_loss_til.item(), input.size(0))
            til_top1_t_m.update(til_acc1.item(), til_output.size(0))
            til_top5_t_m.update(til_acc5.item(), til_output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if last_batch or batch_idx % args.log_interval == 0:
                log_name = f'TestT {til_task}' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2} ({3:>3.0f}%)]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'CIL_Loss_t: {cil_loss_t.avg:>6.4f}  '
                    'CIL_Acc_t@1: {cil_top1t.avg:>7.4f}  '
                    'TIL_Loss_t: {til_loss_t.avg:>6.4f}  '
                    'TIL_Acc_t@1: {til_top1t.avg:>7.4f}'.format(
                        log_name, batch_idx, last_idx, 100. * batch_idx / last_idx,
                        batch_time=batch_time_m,
                        cil_loss_t=cil_losses_t_m, til_loss_t=til_losses_t_m,
                        cil_top1t=cil_top1_t_m, til_top1t=til_top1_t_m))

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, label, _) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input, label = input.cuda(), label.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                _, cil_output, _, _, _ = model(input, task=cil_task)
                til_output, _, _, _, _ = model(input, task=til_task)
            if isinstance(cil_output, (tuple, list)): cil_output = cil_output[0]
            if isinstance(til_output, (tuple, list)): til_output = til_output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                cil_output = cil_output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                til_output = til_output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                label = label[0:label.size(0):reduce_factor]

            loss_cil = loss_fn(cil_output, label)
            loss_til = loss_fn(til_output, label - (args.num_classes // args.tasks * til_task))
            cil_acc1, cil_acc5 = utils.accuracy(cil_output, label, topk=(1, 5))
            til_acc1, til_acc5 = utils.accuracy(til_output, label - (args.num_classes // args.tasks * til_task), topk=(1, 5))

            if args.distributed:
                reduced_loss_cil = utils.reduce_tensor(loss_cil.data, args.world_size)
                cil_acc1 = utils.reduce_tensor(cil_acc1, args.world_size)
                cil_acc5 = utils.reduce_tensor(cil_acc5, args.world_size)
            else:
                reduced_loss_cil = loss_cil.data

            cil_losses_s_m.update(reduced_loss_cil.item(), input.size(0))
            cil_top1_s_m.update(cil_acc1.item(), cil_output.size(0))
            cil_top5_s_m.update(cil_acc5.item(), cil_output.size(0))

            if args.distributed:
                reduced_loss_til = utils.reduce_tensor(loss_til.data, args.world_size)
                til_acc1 = utils.reduce_tensor(til_acc1, args.world_size)
                til_acc5 = utils.reduce_tensor(til_acc5, args.world_size)
            else:
                reduced_loss_til = loss_til.data

            torch.cuda.synchronize()

            til_losses_s_m.update(reduced_loss_til.item(), input.size(0))
            til_top1_s_m.update(til_acc1.item(), til_output.size(0))
            til_top5_s_m.update(til_acc5.item(), til_output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if last_batch or batch_idx % args.log_interval == 0:
                log_name = f'TestS {til_task}'  + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2} ({3:>3.0f}%)]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'CIL_Loss_s: {cil_loss_s.avg:>6.4f}  '
                    'CIL_Acc_s@1: {cil_top1s.avg:>7.4f}  '
                    'TIL_Loss_s: {til_loss_s.avg:>6.4f}  '
                    'TIL_Acc_s@1: {til_top1s.avg:>7.4f}  '
                    'CIL_Loss_t: {cil_loss_t.avg:>6.4f}  '
                    'CIL_Acc_t@1: {cil_top1t.avg:>7.4f}  '
                    'TIL_Loss_t: {til_loss_t.avg:>6.4f}  '
                    'TIL_Acc_t@1: {til_top1t.avg:>7.4f}'.format(
                        log_name, batch_idx, last_idx, 100. * batch_idx / last_idx,
                        batch_time=batch_time_m,
                        cil_loss_s=cil_losses_s_m, cil_loss_t=cil_losses_t_m,
                        til_loss_s=til_losses_s_m, til_loss_t=til_losses_t_m,
                        cil_top1s=cil_top1_s_m, cil_top1t=cil_top1_t_m,
                        til_top1s=til_top1_s_m, til_top1t=til_top1_t_m))

    metrics = OrderedDict([('task', cil_task),
            	           ('cil_loss_s', cil_losses_s_m.avg),
                           ('cil_loss_t', cil_losses_t_m.avg),
                           ('til_loss_s', til_losses_s_m.avg),
                           ('til_loss_t', til_losses_t_m.avg),
                           ('cil_top1_s', cil_top1_s_m.avg),
                           ('cil_top5_s', cil_top5_s_m.avg),
                           ('cil_top1_t', cil_top1_t_m.avg),
                           ('cil_top5_t', cil_top5_t_m.avg),
                           ('til_top1_s', til_top1_s_m.avg),
                           ('til_top5_s', til_top5_s_m.avg),
                           ('til_top1_t', til_top1_t_m.avg),
                           ('til_top5_t', til_top5_t_m.avg)])

    return metrics


if __name__ == '__main__':
    main()
