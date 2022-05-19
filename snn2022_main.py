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
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets, transforms
import pandas as pd


from idiot.idiot_engine import replace


class Net(nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        # mat1 and mat2 shapes cannot be multiplied (64x3072 and 784x200)
        if dataset.lower() == 'mnist':
            input_dim = 784
            hidden_dim = 200
            n_hidden = 3
        elif dataset.lower() == 'cifar10':
            input_dim = 3072
            hidden_dim = 2000
            n_hidden = 6
        else:
            raise ValueError(f"Invalid dataset {dataset}")

        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(n_hidden):
            dropout = nn.Dropout(0.25)
            if i == 0:
                i_dim = input_dim
                o_dim = hidden_dim
            else:
                i_dim = hidden_dim
                o_dim = hidden_dim
            self.linears.append(nn.Linear(i_dim, o_dim))
            self.dropouts.append(dropout)

        self.classifier = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        for linear, dropout in zip(self.linears, self.dropouts):
            x = linear(x)
            x = F.relu(x)
            x = dropout(x)

        x = self.classifier(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
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
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
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
    elif args.dataset.lower() == 'cifar10':
        mean_std = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        initializer = datasets.CIFAR10

    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)
        ]
    )

    train_dataset = initializer(
        '../data', train=True, download=True, transform=transform
    )
    test_dataset = initializer(
        '../data', train=False, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Net(args.dataset).to(device)

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

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        if os.path.exists(checkpoint_path):
            warnings.warn(f"Overwriting existing checkpoint {checkpoint_path}")
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    main()
