"""
MNIST:
    python snn2022_main.py --no-cuda --dataset mnist --lr 0.001 --epochs 10
    ...
    Test set: Average loss: 0.0577, Accuracy: 9822/10000 (98%)

CIFAR10:
    python snn2022_main.py --no-cuda --dataset cifar10 --lr 0.001 --epochs 100 --weight-decay 0.00001  # noqa: E501
    ...
    Test set: Average loss: 1.1823, Accuracy: 5814/10000 (58%)
"""


from __future__ import print_function
import argparse
import warnings
import os
import functools
from collections import defaultdict
from itertools import product


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from torchvision import datasets, transforms
import pandas as pd


from idiot.idiot_engine import replace


class Net(nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        if dataset.lower() == 'mnist':
            input_dim = 784
            n_hidden = 4
            hidden_dims = [200] * n_hidden
            dropout_p = 0.0
        elif dataset.lower() == 'cifar10':
            input_dim = 3072
            n_hidden = 6
            hidden_dims = [2000] * n_hidden
            hidden_dims += [100]
            dropout_p = 0.10
        else:
            raise ValueError(f"Invalid dataset {dataset}")

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                # Input layer
                i_dim = input_dim
                o_dim = hidden_dim
                bn = nn.BatchNorm1d(o_dim)
                dropout = nn.Dropout(dropout_p)
            elif i == len(hidden_dims) - 1:
                # Output layer
                i_dim = hidden_dims[i - 1]
                o_dim = 10
                bn = None
                dropout = None
            else:
                # Hidden layer
                i_dim = hidden_dims[i - 1]
                o_dim = hidden_dim
                bn = nn.BatchNorm1d(o_dim)
                dropout = nn.Dropout(dropout_p)
            self.linears.append(nn.Linear(i_dim, o_dim))
            self.dropouts.append(dropout)
            self.bns.append(bn)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        layer_iter = enumerate(zip(self.linears, self.bns, self.dropouts))
        for i, (linear, bn, dropout) in layer_iter:
            x = linear(x)
            if bn is not None and dropout is not None:
                x = bn(x)
                x = F.relu(x)
                x = dropout(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def count_linear_layers(model):
    n_linear = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            n_linear += 1
    return n_linear


def train(args, model, device, train_loader, optimizer, epoch, scheduler):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        current_lr = get_lr(optimizer)
        if batch_idx % args.log_interval == 0:
            fmt_str = \
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} LR: {:.05f}'
            print(fmt_str.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                current_lr
            ))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    preds = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            preds.append(pred)
            targets.append(target)

    test_loss /= len(test_loader.dataset)

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    accuracy = 100 * accuracy_score(targets, preds)
    correct = accuracy_score(targets, preds, normalize=False)
    cmatrix = confusion_matrix(targets, preds)
    per_class_accuracy = 100 * cmatrix.diagonal() / cmatrix.sum(axis=1)
    per_class_accuracy = [float(f"{acc:.1f}") for acc in per_class_accuracy]

    n = len(test_loader.dataset)
    #accuracy = 100. * correct / len(test_loader.dataset)
    fmt_str = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
    print(fmt_str.format(test_loss, correct, n, accuracy))
    print(f"          Per-class Accuracy: {per_class_accuracy}")

    return test_loss, correct, n, accuracy, per_class_accuracy, targets


def eval_with_lut(
    args, checkpoint_path, model, train_loader, test_loader, device
):
    def get_layer_outputs(module, inputs, outputs, accumulator=None):
        """Save outputs of a layer to a list.
        """
        accumulator.append(outputs)

    def register_layer_output_hooks():
        """Register hooks to accumulate outputs of layers.
        """
        accumulators = defaultdict(list)
        handles = {}
        for module_name, module in model.named_modules():
            if not len(module_name):
                print(f"SKIPPING module name {module_name}")
                continue
            print(f"module name {module_name}")
            if isinstance(module, torch.nn.Linear):
                handles[module_name] = module.register_forward_hook(
                    functools.partial(
                        get_layer_outputs,
                        accumulator=accumulators[module_name]
                    )
                )
        return accumulators, handles

    max_collect_samples = [50000]
    ncodebooks = [-4]
    objectives = ["mse"]

    n_linear = count_linear_layers(model)
    linear_layer_indices = list(range(n_linear)) + [None]

    params = product(
        max_collect_samples,
        ncodebooks,
        objectives,
        linear_layer_indices
    )

    results = []

    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    print("before replacing")
    print(model)

    if args.record_and_save_layer_outputs:
        accumulators, handles = register_layer_output_hooks()

    loss, num_correct, n, accuracy, per_class_accuracy, targets = test(
        model, device, test_loader
    )

    def save_layer_outputs(accumulators):
        """Save lists of tensors in dictionary to a file.
        """
        for module_name in accumulators.keys():
            layer_outputs_list = accumulators[module_name]
            layer_outputs = torch.cat(layer_outputs_list, dim=0)
            torch.save(
                layer_outputs,
                f"layer-outputs-{args.dataset}-{module_name}.pt"
            )

    if args.record_and_save_layer_outputs:
        for handle in handles.values():
            handle.remove()

        save_layer_outputs(accumulators)
        torch.save(targets, f"test-targets-{args.dataset}.pt")

    actual_ncodebooks = [
        "n/a" for _ in model.modules() if isinstance(_, torch.nn.Linear)
    ]
    results.append((
        actual_ncodebooks,
        actual_ncodebooks,
        "N/A",
        "N/A",
        "N/A",
        loss,
        accuracy
    ))

    def get_actual_ncodebooks(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                try:
                    yield module._idiot_opts['actual_ncodebooks']
                except KeyError:
                    yield "n/a"

    for mcs, ncb, obj, linear_module_index in params:
        # Reload model before each replacement.
        model = Net(args.dataset).to(device)
        model.load_state_dict(state_dict)
        if linear_module_index is None:
            layer_indices = None
        else:
            layer_indices = [linear_module_index]
        model = replace(
            train_loader,
            model,
            device,
            layer_indices=layer_indices,
            ncodebooks=ncb,
            max_collect_samples=mcs,
            objective=obj,
            force_softmax_and_kld_on_output_layer=True
        )
        print(f"after replacing linear layer {linear_module_index}")
        print(model)
        loss, num_correct, n, accuracy, per_class_accuracy, _ = test(
            model, device, test_loader
        )
        actual_ncodebooks = list(get_actual_ncodebooks(model))
        results.append((
            ncodebooks,
            actual_ncodebooks,
            mcs,
            obj,
            linear_module_index,
            loss,
            accuracy
        ))

    for res in results:
        print(res)

    columns = [
        "ncodebooks",
        "actual_ncodebooks",
        "max_collect_samples",
        "objective",
        "linear_module_index",
        "loss",
        "accuracy"
    ]

    def save_data_frame():
        df = pd.DataFrame(data=results, columns=columns)
        df.to_csv("results.csv", index=False)

    save_data_frame()


def do_train(
    args, model, train_loader, test_loader, device, optimizer, scheduler
):
    for epoch in range(1, args.epochs + 1):
        train(
            args,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            scheduler
        )
        test(model, device, test_loader)


def do_save(args, model, checkpoint_path):
    if args.save_model:
        if os.path.exists(checkpoint_path):
            warnings.warn(f"Overwriting existing checkpoint {checkpoint_path}")
        torch.save(model.state_dict(), checkpoint_path)


def finetune_with_lut(
    args, checkpoint_path, model, train_loader, test_loader, device, **config
):
    # Set layer L=1, M=num layers
    # While L <= M:
    #   Replace layer L of network with specified vector quantization config
    #   Fix the weights of layers 1 .. L and fine-tune the other layers
    #   L = L + 1
    # n_linear = count_linear_layers(model)

    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)

    # Disable dropout when fine-tuning. We don't want the lookup tables to be
    # fitted using inputs with stochastic error.
    for module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    print("=> Original model")
    print(model)
    test(model, device, test_loader)

    # Set layer_indices to an empty list and run fit_lut() ourselves.
    model = replace(
        train_loader,
        model,
        device,
        layer_indices=[],
        **config
    )

    linear_modules = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_modules.append((module_name, module))

    iterator = enumerate(linear_modules)
    for linear_module_index, (module_name, module) in iterator:
        print(f"=> Replacing layer {module_name}")
        layer_input = []
        module._idiot_phase = "collect_input"
        module._idiot_input = layer_input
        # Collect input for this linear layer.
        for data, label in train_loader:
            len_before = len(layer_input)
            model(data)
            len_after = len(layer_input)
            if len_after == len_before:
                # The length of layer_input didn't change this time, so we're
                # done collecting input.
                break
        idiot_input_concat = torch.cat(layer_input, dim=0)

        # Fit the lookup table for this linear layer.
        module.fit_lut(idiot_input_concat, None)
        module._idiot_phase = "apply_lut"

        # Report test set accuracy before fine-tuning the network with this
        # (and all previous) linear layer's matrix multiplications replaced by
        # a fixed LUT.
        print(f"=> Before fine-tuning layers after {module_name}")
        print(model)
        test(model, device, test_loader)

        """
        If a module is BatchNorm1d, include it in trainable params.

        If a module is Linear, only include it in trainable params if it is
        later in the network than the layer that we just replaced earlier in
        the above loop.
        """
        def get_trainable_params():
            trainable_params = []

            iterator = enumerate(model.linears.named_children())
            for i, (linear_name, linear_module) in iterator:
                # i = [0, 1, 2, 3]
                # linear_module_index = [0, 1, 2, 3]
                if i > linear_module_index:
                    # When training a linear layer, also train its previous
                    # batch norm layer.
                    bn_index = int(linear_name) - 1
                    bn_module = model.bns[bn_index]
                    if bn_module is not None:
                        print(f"=> Fine-tuning bns.{bn_index} parameters")
                        trainable_params.append(
                            {'params': bn_module.parameters()}
                        )

                    print(f"=> Fine-tuning linears.{linear_name} parameters")
                    trainable_params.append(
                        {'params': linear_module.parameters()}
                    )

            return trainable_params

        trainable_params = get_trainable_params()
        if len(trainable_params) == 0:
            print("=> Fine-tuning procedure is complete")
            test(model, device, test_loader)
            break

        optimizer = optim.SGD(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        do_train(
            args,
            model,
            train_loader,
            test_loader,
            device,
            optimizer,
            scheduler
        )

    output_checkpoint_path = f"{args.dataset}_mlp_finetuned.pt"
    do_save(args, model, output_checkpoint_path)


def train_from_scratch(
    args, checkpoint_path, model, train_loader, test_loader, device
):
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    do_train(
        args,
        model,
        train_loader,
        test_loader,
        device,
        optimizer,
        scheduler
    )
    do_save(args, model, checkpoint_path)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument(
        '--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
        required=True,
        help='Dataset to train/test on'
    )
    parser.add_argument(
        '--eval-with-lut', action='store_true',
        help='Evaluate with lookup tables'
    )
    parser.add_argument(
        '--finetune-with-lut', action='store_true',
        help='Finetune a trained checkpoint, layerwise, with lookup tables'
    )
    parser.add_argument(
        '--record-and-save-layer-outputs', action='store_true',
        help=(
            "Record outputs of each Linear layer of original model. "
            "This argument only has an effect with --eval-with-lut. "
            "Outputs are saved to files layer-outputs-DATASET-LAYER_NAME.pt."
        )
    )
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--test-batch-size', type=int, default=1000, metavar='N',
        help='input batch size for testing (default: 1000)'
    )
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument(
        '--weight-initializer', type=str, default="xavier_normal_",
        choices=[
            "orthogonal_", "xavier_uniform_", "xavier_normal_",
            "kaiming_uniform_", "kaiming_normal_"
        ],
        help="torch.nn.init initializer to use for weights"
    )
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before logging training status'
    )
    parser.add_argument(
        '--save-model', action='store_true', dest="save_model", default=True,
        help='Save the model after training'
    )
    parser.add_argument(
        '--no-save-model', action='store_false', dest="save_model",
        help='Do not save the model after training'
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.dataset.lower() == 'mnist':
        mean_std = ((0.1307,), (0.3081,))
        initializer = datasets.MNIST
        # 98% accuracy on MNIST without augmentations.
        train_xforms = []
    elif args.dataset.lower() == 'cifar10':
        mean_std = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        train_xforms = [
            transforms.Pad(4),
            transforms.RandomCrop(32)
        ]
        initializer = datasets.CIFAR10

    common_xforms = [
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ]

    train_transform = transforms.Compose(
        [
            *common_xforms,
            *train_xforms
        ]
    )
    train_dataset = initializer(
        '../data', train=True, download=True, transform=train_transform
    )

    test_transform = transforms.Compose([*common_xforms])
    test_dataset = initializer(
        '../data', train=False, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Net(args.dataset).to(device)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            weight_initializer_ = getattr(
                torch.nn.init, args.weight_initializer
            )
            if "kaiming" in args.weight_initializer:
                weight_initializer_kwargs = {"non_linearity": "relu"}
            else:
                gain = torch.nn.init.calculate_gain("relu")
                weight_initializer_kwargs = {"gain": gain}
            weight_initializer_(
                module.weight.data, **weight_initializer_kwargs
            )
            torch.nn.init.constant_(module.bias.data, 0.0)

    checkpoint_path = f"{args.dataset}_mlp.pt"

    if args.eval_with_lut:
        eval_with_lut(
            args, checkpoint_path, model, train_loader, test_loader, device
        )
        return
    elif args.finetune_with_lut:
        config = dict(
            ncodebooks=-4,
            max_collect_samples=3000,
            objective="mse",
            force_softmax_and_kld_on_output_layer=False
        )
        finetune_with_lut(
            args,
            checkpoint_path,
            model,
            train_loader,
            test_loader,
            device,
            **config
        )
    else:
        train_from_scratch(
            args, checkpoint_path, model, train_loader, test_loader, device
        )


if __name__ == '__main__':
    main()
