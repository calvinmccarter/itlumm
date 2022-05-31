# coding: utf-8
"""
Estimate the coding rate distortion of layer output.

    python estimate_layer_output_distortion.py \
        --layer-output-paths layer-outputs-mnist-* \
        --test-target-path test-targets-mnist.pt

Yu, Y., Chan, K. H. R., You, C., Song, C., & Ma, Y. (2020).
Learning diverse and discriminative representations via the principle of
maximal coding rate reduction. NeurIPS, 33, 9422-9434.
"""

from argparse import ArgumentParser

import torch


def estimate_rate_distortion(layer_output, eps=1.):
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
    distortion = torch.logdet(I + scalar * Z.T @ Z)
    # Convert from nats to bits
    distortion /= 2.
    return distortion


def estimate_conditional_rate_distortion(layer_output, targets, eps=1.):
    """Estimate the conditional rate distortion, the number of binary
    bits needed to encode a sample such that expected decoding
    error < eps (where eps is the desired precision).
    This is the conditional distortion, conditioned on class labels
    for each sample.

    Arguments:
        layer_output (torch.Tensor): output of layer
        targets (torch.Tensor): labels of samples -- 1d of ints
        eps (float): maximum allowed distortion

    Returns:
        distortion (float): rate distortion of layer outputs

    """
    Z = layer_output
    labels = targets
    assert Z.dim() == 2
    N, D = Z.shape
    I = torch.eye(D)
    distortion = torch.tensor(0.)
    for uval in labels.unique():
        n_uval = (labels == uval).sum()
        scalar = D / (n_uval * eps * eps)
        distortion = distortion + n_uval/(2*N) * torch.logdet(
            I + scalar * (Z[uval==labels, :]).T @ Z[uval==labels, :])
    return distortion


def main(layer_output_paths, test_target_path):
    test_targets = torch.load(test_target_path)
    for layer_output_path in layer_output_paths:
        layer_output = torch.load(layer_output_path)
        distortion = estimate_rate_distortion(layer_output)
        cdistortion = estimate_conditional_rate_distortion(
            layer_output, test_targets)
        print(
            f"{layer_output_path} combined:{distortion:.3f} "
            f"conditional:{cdistortion:.3f} delta:{distortion-cdistortion:.3f}"
        )


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
    parser.add_argument(
        "--test-target-path",
        default="",
        required=True,
        type=str,
        help="Path to saved torch tensor containing test set targets"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.layer_output_paths, args.test_target_path)
