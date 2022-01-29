# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

from idiot.idiot import (
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
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
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
def replace(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Replace:'

    # switch to evaluation mode
    model.eval()

    idiot_ordering = []  # ordered list of IdiotLinear layers
    max_collect_samples = 10240
    algorithm = "pluto"
    idiot_opts = {
        "max_collect_samples": max_collect_samples,
        "ncodebooks": None,
        "nonzeros_heuristic": "opq",
        "algorithm": algorithm,
        "objective": "mse",
        "accumulate_how": "mean",
    }


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

    return model

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
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
def replace(data_loader, net, device):
    net.eval()

    idiot_ordering = []  # ordered list of IdiotLinear layers
    max_collect_samples = 10240
    algorithm = "pluto"
    idiot_opts = {
        "max_collect_samples": max_collect_samples,
        "ncodebooks": -2,
        "nonzeros_heuristic": "r2",
        "algorithm": algorithm,
        "objective": "mse",
        "accumulate_how": "mean",
    }

    new_net = replace_descendants(
        net,
        idiot_ordering,
        idiot_opts,
        "",
        None,
    )
    new_net.eval()

    set_all_descendant_attrs(new_net, "_idiot_phase", "find_ordering")
    for data, label in test_loader:
        output = new_net(data) # mutates idiot_ordering
        break
    set_all_descendant_attrs(new_net, "_idiot_phase", "noop")

    def f_softmax(x):
        return torch.softmax(x, dim=1)
    get_descendant(
        new_net, idiot_ordering[-1])._idiot_activation = f_softmax
    def set_activation_gelu(mod):
        if isinstance(mod, IdiotLinear):
            if mod._idiot_name.endswith("fc1"):
                print(mod._idiot_name)
                mod._idiot_activation = F.gelu
    new_net.apply(set_activation_gelu)
    exit(0)

    # PLUTO
    for lname in idiot_ordering:
        acc = 0.0
        for data, label in test_loader:
            output = new_net(data)
            acc += (output.argmax(dim=1) == label).float().mean()
        acc = acc / len(test_loader)
        print(f"idiot-{algorithm}: before replacing {lname}: acc={acc}")

        idiot_input = []  # list for storing all activations
        get_descendant(new_net, lname)._idiot_phase = "collect_input"
        get_descendant(new_net, lname)._idiot_input = idiot_input
        for bix, (data, label) in enumerate(train_loader):
            # Modifies idiot_input
            idiot_input_len = len(idiot_input)
            output = new_net(data)
            if len(idiot_input) == idiot_input_len:
                break

        idiot_input_concat = torch.cat(idiot_input, dim=0)
        get_descendant(new_net, lname).fit_lut(idiot_input_concat, None)
        get_descendant(new_net, lname)._idiot_phase = "apply_lut"

    acc = 0.0
    for data, label in test_loader:
        # Modifies idiot_input
        output = new_net(data)
        acc += (output.argmax(dim=1) == label).float().mean()
    acc = acc / len(test_loader)
    print(f"idiot-{algorithm}: final {max_collect_samples}: acc={acc}")

    return model


