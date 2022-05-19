# coding: utf-8
"""
Estimate the instrinc dimensionality of layer output using TwoNN estimator.

    python estimate_layer_output_dimensionality.py \
        --layer-output-paths \
            layer-outputs-mnist-linears.* layer-outputs-mnist-classifier.pt
"""

from argparse import ArgumentParser

import torch
import skdim


def estimate_intrinsic_dimensionality(layer_output):
    """Estimate the intrinsic dimensionality of layer output using TwoNN.

    Arguments:
        layer_output (torch.Tensor): output of layer

    Returns:
        dimensionality (float): estimated intrinsic dimensionality of layer

    """
    estimator = skdim.id.TwoNN()
    estimator.fit(layer_output.numpy())
    dimensionality = estimator.dimension_
    return dimensionality


def main(layer_output_paths):
    for layer_output_path in layer_output_paths:
        layer_output = torch.load(layer_output_path)
        dimensionality = estimate_intrinsic_dimensionality(layer_output)
        print(layer_output_path, dimensionality)


def get_parser():
    parser = ArgumentParser(
        description="Estimate dimensionality of layer outputs"
    )
    parser.add_argument(
        "--layer-output-paths",
        default=[],
        nargs="+",
        required=True,
        type=str,
        help="Path to saved torch tensor containing layer outputs"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.layer_output_paths)
