import yaml
import sys
import argparse
import torch
import cvnets
from cvnets.models import detection
import collections
from headutils import * 

print('Finished Imports')



def flatten_yaml_as_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def std_cls_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser

    group.add_argument('--model.classification.classifier-dropout', type=float, default=0.0,
                       help="Dropout rate in classifier")
    # setattr(group,'--model.classificaiton.classifier-dropout')
    group.add_argument('--model.classification.name', type=str, default="mobilenetv2", help="Model name")
    group.add_argument('--model.classification.n-classes', type=int, default=1000,
                       help="Number of classes in the dataset")
    group.add_argument('--model.classification.pretrained', type=str, default=None,
                       help="Path of the pretrained backbone")
    group.add_argument('--model.classification.freeze-batch-norm', action="store_true", help="Freeze batch norm layers")

    group.add_argument('--model.classification.activation.name', default=None, type=str,
                       help='Non-linear function type')
    group.add_argument('--model.classification.activation.inplace', action='store_true',
                       help='Inplace non-linear functions')
    group.add_argument('--model.classification.activation.neg-slope', default=0.1, type=float,
                       help='Negative slope in leaky relu')
    group.add_argument('--model.classification.mit.mode', type=str, default=None,
                       choices=['xx_small', 'x_small', 'small'], help="MIT mode")
    group.add_argument('--model.classification.mit.attn-dropout', type=float, default=0.1,
                       help="Dropout in attention layer")
    group.add_argument('--model.classification.mit.ffn-dropout', type=float, default=0.0,
                       help="Dropout between FFN layers")
    group.add_argument('--model.classification.mit.dropout', type=float, default=0.1,
                       help="Dropout in Transformer layer")
    group.add_argument('--model.classification.mit.transformer-norm-layer', type=str, default="layer_norm",
                       help="Normalization layer in transformer")
    group.add_argument('--model.classification.mit.no-fuse-local-global-features', action="store_true",
                       help="Do not combine local and global features in MIT block")
    group.add_argument('--model.classification.mit.conv-kernel-size', type=int, default=3,
                       help="Kernel size of Conv layers in MIT block")

    group.add_argument('--model.classification.mit.head-dim', type=int, default=None,
                       help="Head dimension in transformer")
    group.add_argument('--model.classification.mit.number-heads', type=int, default=None,
                       help="No. of heads in transformer")
    group.add_argument('--model.activation.name', default='relu', type=str, help='Non-linear function type')
    group.add_argument('--model.activation.inplace', action='store_true', help='Inplace non-linear functions')
    group.add_argument('--model.activation.neg-slope', default=0.1, type=float, help='Negative slope in leaky relu')
    group.add_argument('--model.normalization.name', default='batch_norm', type=str, help='Normalization layer')
    group.add_argument('--model.normalization.groups', default=32, type=str,
                       help='Number of groups in group normalization layer')
    group.add_argument("--model.normalization.momentum", default=0.1, type=float,
                       help='Momentum in normalization layers')
    group.add_argument("--model.layer.conv_init", default='kaiming_normal', type=str)
    group.add_argument("--model.layer.linear_init", default='trunc_normal', type=str)
    group.add_argument("--model.layer.linear_init_std_dev", default=0.02, type=float)
    group.add_argument("--model.layer.global_pool", default='mean', type=str)

    # group.add_argument("--evaluation.detection.save-overlay-boxes", action="store_true",
    #                    help="enable this flag to visualize predicted masks on top of input image")
    # group.add_argument("--evaluation.detection.mode", type=str, default="validation_set", required=True,
    #                    choices=["single_image", "image_folder", "validation_set"],
    #                    help="Contribution of mask when overlaying on top of RGB image. ")
    # group.add_argument("--evaluation.detection.path", type=str, default=None,
    #                    help="Path of the image or image folder (only required for single_image and image_folder modes)")
    # group.add_argument("--evaluation.detection.num-classes", type=str, default=None,
    #                    help="Number of segmentation classes used during training")
    # group.add_argument("--evaluation.detection.resize-input-images", action="store_true", default=False,
    #                    help="Resize the input images")

    group.add_argument('--model.detection.name', type=str, default=None, help="Model name")
    group.add_argument('--model.detection.n-classes', type=int, default=2, help="Number of classes in the dataset")
    group.add_argument('--model.detection.pretrained', type=str, default=None,
                       help="Path of the pretrained model")
    group.add_argument('--model.detection.output-stride', type=int, default=None,
                       help="Output stride in classification network")
    group.add_argument('--model.detection.replace-stride-with-dilation', action="store_true",
                       help="Replace stride with dilation")
    group.add_argument('--model.detection.freeze-batch-norm', action="store_true",
                       help="Freeze batch norm layers")
    group.add_argument('--ddp.enable', action="store_true", default=False, help="Use DDP")
    group.add_argument('--ddp.rank', type=int, default=0, help="Node rank for distributed training")
    group.add_argument('--ddp.world-size', type=int, default=1, help="World size for DDP")
    group.add_argument('--ddp.dist-url', type=str, default=None, help="DDP URL")
    group.add_argument('--ddp.dist-port', type=int, default=6006, help="DDP Port")
    group.add_argument('--dataset.name', default='hollyhead_ssd')
    group.add_argument('--dataset.category', default = 'detection')
    group.add_argument('--dataset.root_train',default = '.')
    group.add_argument('--dataset.root_val',default = '.')
    return group




def load_config_file(opts, opts_file):
    with open(opts_file) as yaml_file:
        # model_opts = yaml.load(file, Loader=yaml.FullLoader)
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

        flat_cfg = flatten_yaml_as_dict(cfg)
        # flat_cfg = {'--'+key:flat_cfg[key] for key in flat_cfg.keys()}
        for k, v in flat_cfg.items():
            print(k, v, hasattr(opts, k))
            if hasattr(opts, k):
                setattr(opts, k, v)
    return opts


def generate_model(opts_file):
    opts = argparse.ArgumentParser(description='Model arguments', add_help=True)
    new_opts = std_cls_model_args(opts)
    opts = new_opts.parse_args(args=[])
    opts = load_config_file(opts,opts_file)
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = detection.build_detection_model(opts)
    return model

print('Got Passed Functions')


if __name__ == '__main__':
    sys.path.insert(0,'.')
    print('here')
    model = generate_model('config/detection/ssd_mobilevit_small_hollyhead.yaml')
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = model(torch.rand(1,3,320,320).to(device))
    print(out[0].shape,out[1].shape)
    print(len(model.priors_cxcy))
    print([model.priors_cxcy[i].shape for i in range(len(model.priors_cxcy))])
    # opts = argparse.ArgumentParser(description='Model arguments', add_help=True)
    # new_opts = std_cls_model_args(opts)
    # opts = opts.parse_args(args=[])
    # opts = load_config_file(opts)
    #
    #
    # device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = detection.build_detection_model(opts)
    # model.to(device)


