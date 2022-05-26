"""
MNIST:
    python snn2022_main.py --no-cuda --dataset mnist --lr 0.001 --epochs 10
    ...
    Test set: Average loss: 0.0577, Accuracy: 9822/10000 (98%)

CIFAR10:
    python snn2022_main.py --no-cuda --dataset cifar10 --lr 0.001 --epochs 100 --weight-decay 0.00001
    ...
    Test set: Average loss: 1.1823, Accuracy: 5814/10000 (58%)
"""


from __future__ import print_function
import argparse
import warnings
import os
import functools
from collections import defaultdict


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
        # mat1 and mat2 shapes cannot be multiplied (64x3072 and 784x200)
        if dataset.lower() == 'mnist':
            input_dim = 784
            n_hidden = 3
            hidden_dims = [200] * n_hidden
            dropout_p = 0.25
        elif dataset.lower() == 'cifar10':
            input_dim = 3072
            n_hidden = 5
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
                i_dim = input_dim
                o_dim = hidden_dim
            else:
                i_dim = hidden_dims[i - 1]
                o_dim = hidden_dim
            self.linears.append(nn.Linear(i_dim, o_dim))
            self.bns.append(nn.BatchNorm1d(o_dim))
            self.dropouts.append(nn.Dropout(dropout_p))

        self.classifier = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        for linear, bn, dropout in zip(self.linears, self.bns, self.dropouts):
            x = linear(x)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)

        x = self.classifier(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} LR: {:.05f}'.format(
                epoch,batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                current_lr
            ))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    n = len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, n, accuracy
    ))

    return test_loss, correct, n, accuracy


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
        '--record-and-save-layer-outputs', action='store_true',
        help=(
            "Record outputs of each Linear layer of original model. "
            "This argument only has an effect with --eval-with-lut. "
            "Outputs are saved to files layer-outputs-DATASET-LAYER_NAME.pt."
        )
    )
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--weight-initializer', type=str, default="xavier_normal_",
                        choices=[
                            "orthogonal_", "xavier_uniform_", "xavier_normal_",
                            "kaiming_uniform_", "kaiming_normal_"
                        ],
                        help="torch.nn.init initializer to use for weights")
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', dest="save_model", default=True,
                        help='Save the model after training')
    parser.add_argument('--no-save-model', action='store_false', dest="save_model",
                        help='Do not save the model after training')
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

    val_transform = transforms.Compose([*common_xforms])
    test_dataset = initializer(
        '../data', train=False, transform=val_transform
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

    def get_actual_ncodebooks(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                try:
                    yield module._idiot_opts['actual_ncodebooks']
                except KeyError:
                    yield "n/a"

    if args.eval_with_lut:
        from itertools import product

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
                if isinstance(module, torch.nn.Linear):
                    handles[module_name] = module.register_forward_hook(
                        functools.partial(
                            get_layer_outputs,
                            accumulator=accumulators[module_name]
                        )
                    )
            return accumulators, handles


        max_collect_samples = [300]
        #ncodebooks = [-1, -2, -3, -4, -8, -16]
        ncodebooks = [-2, -4, -8, -16]
        objectives = ["mse"]

        n_linear = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                n_linear += 1
        linear_layer_indices = range(n_linear)
        
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

        loss, num_correct, n, accuracy = test(model, device, test_loader)

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

        actual_ncodebooks = ["n/a"
            for _ in model.modules() if isinstance(_, torch.nn.Linear)
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

        for mcs, ncb, obj, linear_layer_index in params:
            # Reload model before each replacement.
            model = Net(args.dataset).to(device)
            model.load_state_dict(state_dict)
            model = replace(
                train_loader,
                model,
                device,
                layer_indices=[linear_layer_index],
                ncodebooks=ncb,
                max_collect_samples=mcs,
                objective=obj,
                force_softmax_and_kld_on_output_layer=True
            )
            print(f"after replacing linear layer {linear_layer_index}")
            print(model)
            loss, num_correct, n, accuracy = test(model, device, test_loader)
            actual_ncodebooks = list(get_actual_ncodebooks(model))
            results.append((
                ncodebooks,
                actual_ncodebooks,
                mcs,
                obj,
                linear_layer_index,
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
            "linear_layer_index",
            "loss",
            "accuracy"
        ]

        def save_data_frame():
            df = pd.DataFrame(data=results, columns=columns)
            df.to_csv("results.csv", index=False)


        save_data_frame()


        return

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

    if args.save_model:
        if os.path.exists(checkpoint_path):
            warnings.warn(f"Overwriting existing checkpoint {checkpoint_path}")
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    main()
