import os
import pickle
import torch
import torch.multiprocessing as tmp
import torchvision as tv
import torch.nn.functional as F
import torchvision.transforms as tvt

from PIL import Image
from torchvision import transforms

from idiot.idiot import (
    IdiotLinear,
    get_descendant,
    replace_descendants,
    set_all_descendant_attrs,
)

from idiot.mlp_mixer import MLPMixer

home = os.environ['HOME']

if __name__ == "__main__":
    transform = tvt.Compose([
        tvt.ToTensor(),
        tvt.Resize((32, 32)),
        #tvt.RandomHorizontalFlip(),
        tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
    train_data = tv.datasets.CIFAR10(
        './',
        train=True,
        transform=transform,
        target_transform=None,
        download=True,
    )
    test_data = tv.datasets.CIFAR10(
        './',
        train=False,
        transform=transform,
        target_transform=None,
        download=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=8,
        shuffle=False,
        num_workers=1,
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        num_workers=1,
    )

    tmp.freeze_support()

    net = MLPMixer(
        image_shape=(32, 32), 
        patch_size=8,
        num_channels=3, 
        num_hidden_dim=128,
        num_layers=6,
        num_classes=10,
        dropout=0.2,
        mlp_dim_factor=2,
    )
    net.load_state_dict(torch.load(
        f"{home}/sandbox/MLP-Mixer/training_artefacts/model_epoch_41.pth"))
    net.eval()

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
                mod._idiot_activation = F.gelu
    new_net.apply(set_activation_gelu)

    # PLUTO
    with torch.no_grad():
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


