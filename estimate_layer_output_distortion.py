# coding: utf-8
"""
Estimate the instrinc dimensionality of layer output using TwoNN estimator.

    python estimate_layer_output_dimensionality.py \
        --layer-output-paths \
            layer-outputs-mnist-linears.* layer-outputs-mnist-classifier.pt
"""

from argparse import ArgumentParser

import torch


def estimate_rate_distortion(layer_output, eps=0.1):
    """Estimate the rate distortion, the number of binary
    bits needed to encode a sample such that expected decoding
    error < eps (where eps is the desired precision).

    Arguments:
        layer_output (torch.Tensor): output of layer

        eps (float): prescribed precision

    Returns:
        distortion (float): rate distortion of layer outputs

    """
    Z = layer_output
    assert Z.dim() == 2
    N, D = Z.shape
    I = torch.eye(D)
    scalar = D / (N * eps * eps)
    distortion = torch.logdet(I + scalar * Z.T @ Z) / 2.
    return distortion


def estimate_conditional_rate_distortion(layer_output, labels, eps=0.1):
    """Estimate the conditional rate distortion, the number of binary
    bits needed to encode a sample such that expected decoding
    error < eps (where eps is the desired precision).
    This is the conditional distortion, conditioned on class labels
    for each sample.

    Arguments:
        layer_output (torch.Tensor): output of layer

        labels (torch.Tensor): labels of samples -- 1d of ints

        eps (float): maximum allowed distortion

    Returns:
        distortion (float): rate distortion of layer outputs

    """
    Z = layer_output
    assert Z.dim() == 2
    N, D = Z.shape
    I = torch.eye(D)
    K = labels.unique().numel()
    distortion = torch.tensor(0.)
    for uval in labels.unique():
        n_uval = (labels == uval).sum()
        scalar = D / (n_uval * eps * eps)
        distortion = distortion + n_uval/(2*N) * torch.logdet(
            I + scalar * (Z[uval==labels, :]).T @ Z[uval==labels, :])
    return distortion


def main(layer_output_paths):
    for layer_output_path in layer_output_paths:
        layer_output = torch.load(layer_output_path)
        distortion = estimate_rate_distortion(layer_output)
        print(layer_output_path, distortion)


def get_parser():
    parser = ArgumentParser(
        description="Estimate rate distortion of layer outputs"
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
