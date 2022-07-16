# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from driveit.losses import DistillationLoss
import driveit.utils

from driveit.driveit import (
    IdiotLinear,
    get_descendant,
    replace_descendants,
    set_all_descendant_attrs,
)



def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = driveit.utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', driveit.utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = driveit.utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def replace(
    data_loader,
    net,
    device,
    layer_indices=None,
    force_softmax_and_kld_on_output_layer=True,
    **driveit_opt_kwargs
):
    net.eval()

    driveit_ordering = []  # ordered list of IdiotLinear layers
    max_collect_samples = 1000
    algorithm = "pluto"
    driveit_opts = {
        "max_collect_samples": max_collect_samples,
        "ncodebooks": -4,
        "nonzeros_heuristic": "r2",
        "algorithm": algorithm,
        "objective": "mse",
        "accumulate_how": "mean",
    }
    driveit_opts.update(**driveit_opt_kwargs)

    new_net = replace_descendants(
        net,
        driveit_ordering,
        driveit_opts,
        "",
        None,
    )
    new_net.eval()

    set_all_descendant_attrs(new_net, "_driveit_phase", "find_ordering")
    for data, label in data_loader:
        output = new_net(data) # mutates driveit_ordering
        break
    set_all_descendant_attrs(new_net, "_driveit_phase", "noop")
    print(driveit_ordering)
    print(new_net)

    if force_softmax_and_kld_on_output_layer:
        # Process the last layer's output with softmax and fit the lookup table
        # using the softmax output. Force the objective to be kld for the last
        # layer. This hard-coded configuration is OK for simple classifier
        # networks that we're considering now.
        def f_softmax(x):
            return torch.softmax(x, dim=1)
        output_layer = get_descendant(new_net, driveit_ordering[-1])
        output_layer._driveit_activation = f_softmax
        output_layer._driveit_opts["objective"] = "kld"

    if layer_indices is None:
        layer_indices = set(range(len(driveit_ordering)))

    # PLUTO
    for i, lname in enumerate(driveit_ordering):
        if i not in layer_indices:
            continue
        print(f"replacing {i}-th layer {lname}")
        driveit_input = []  # list for storing all activations
        get_descendant(new_net, lname)._driveit_phase = "collect_input"
        get_descendant(new_net, lname)._driveit_input = driveit_input
        for bix, (data, label) in enumerate(data_loader):
            # Modifies driveit_input
            driveit_input_len = len(driveit_input)
            output = new_net(data)
            if len(driveit_input) == driveit_input_len:
                break

        driveit_input_concat = torch.cat(driveit_input, dim=0)
        get_descendant(new_net, lname).fit_lut(driveit_input_concat, None)
        get_descendant(new_net, lname)._driveit_phase = "apply_lut"

    return new_net
