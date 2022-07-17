import os
import pickle
import torch
import torch.multiprocessing as tmp
import torchvision as tv
import torch.nn.functional as F
import torchvision.transforms as tvt

from PIL import Image
from torchvision import transforms

from driveit.driveit import (
    DriveitLinear,
    get_descendant,
    replace_descendants,
    set_all_descendant_attrs,
)

from driveit.mlp_mixer import MLPMixer

home = os.environ['HOME']

if __name__ == "__main__":
    transform = tvt.Compose([
        tvt.ToTensor(),
        tvt.Resize((32, 32)),
        #tvt.RandomHorizontalFlip(),
        tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
    test_data = tv.datasets.CIFAR10(
        './',
        train=False,
        transform=transform,
        target_transform=None,
        download=True,
    )
    with open(f"{home}/sandbox/objective-correlative/cifar-10-batches-py/batches.meta", 'rb') as pickleFile:
        cifar10_classes = pickle.load(pickleFile)['label_names'] 

    filename = f"{home}/sandbox/MLP-Mixer/images/frog.jpg"
    image = Image.open(filename)
    image = transform(image)
    image = image.unsqueeze(0)

    tmp.freeze_support()

    """
    net = MLPMixer(
        image_shape=(32, 32), 
        patch_size=8,
        num_channels=3, 
        num_hidden_dim=32,
        num_layers=3,
        num_classes=10,
        dropout=0.2,
        mlp_dim_factor=2,
    )
    net.load_state_dict(torch.load(
        f"{home}/sandbox/MLP-Mixer/mlp_mixer_cifar10_small.pt"))
    """
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
    output = torch.softmax(net(image).squeeze(), dim = 0)
    values, indices = torch.topk(output, k = 5)
    class_names = [cifar10_classes[i] for i in indices.numpy().astype(int)]
    values = values.detach().numpy()
    print(f"original MLPMixer {class_names} {values}")

    driveit_ordering = []  # ordered list of DriveitLinear layers
    max_collect_samples = 10240
    algorithm = "pluto"
    driveit_opts = {
        "max_collect_samples": max_collect_samples,
        "ncodebooks": None,
        "nonzeros_heuristic": "opq",
        "algorithm": algorithm,
        "objective": "mse",
        "accumulate_how": "mean",
    }

    new_net = replace_descendants(
        net,
        driveit_ordering,
        driveit_opts,
        "",
        None,
    )
    #print(new_net)

    new_net.eval()
    output = torch.softmax(new_net(image).squeeze(), dim = 0)
    values, indices = torch.topk(output, k = 5)
    class_names = [cifar10_classes[i] for i in indices.numpy().astype(int)]
    values = values.detach().numpy()
    print(f"after Linear->DriveitLinear {class_names} {values}")

    set_all_descendant_attrs(new_net, "_driveit_phase", "find_ordering")
    output = new_net(image)  # mutates driveit_ordering
    set_all_descendant_attrs(new_net, "_driveit_phase", "noop")


    def f_softmax(x):
        return torch.softmax(x, dim=1)
    #print(new_net)
    #print(output)
    #print(f_softmax(output))
    #print(get_descendant(new_net, driveit_ordering[-1]))
    get_descendant(
        new_net, driveit_ordering[-1])._driveit_activation = f_softmax
    #print(new_net)
    def set_activation_relu(mod):
        if isinstance(mod, DriveitLinear):
            if mod._driveit_name.endswith("fc1"):
                mod._driveit_activation = F.gelu
    new_net.apply(set_activation_relu)


    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=8,
        shuffle=False,
        num_workers=1,
    )

    # PLUTO
    with torch.no_grad():
        for lname in driveit_ordering:
            driveit_input = []  # list for storing all activations
            get_descendant(new_net, lname)._driveit_phase = "collect_input"
            get_descendant(new_net, lname)._driveit_input = driveit_input
            acc = 0.0
            for data, label in test_loader:
                # Modifies driveit_input_cur and driveit_output_next
                output = new_net(data)
                acc += (output.argmax(dim=1) == label).float().mean()
            acc = acc / len(test_loader)
            print(f"driveit-{algorithm}: before replacing {lname}: acc={acc}")

            driveit_input_concat = torch.cat(driveit_input, dim=0)
            get_descendant(new_net, lname).fit_lut(driveit_input_concat, None)
            get_descendant(new_net, lname)._driveit_phase = "apply_lut"

            output = torch.softmax(new_net(image).squeeze(), dim = 0)
            values, indices = torch.topk(output, k = 5)
            class_names = [cifar10_classes[i] for i in indices.numpy().astype(int)]
            values = values.detach().numpy()
            print(f"after {algorithm} {class_names} {values}")
        acc = 0.0
        for data, label in test_loader:
            # Modifies driveit_input_cur and driveit_output_next
            output = new_net(data)
            acc += (output.argmax(dim=1) == label).float().mean()
        acc = acc / len(test_loader)
        print(f"driveit-{algorithm}: final {max_collect_samples}: acc={acc}")


    # BBPLUTO
    """
    print("\n".join(driveit_ordering))
    with torch.no_grad():
        driveit_output_next = None
        driveit_name_next = None
        for ix_cur, name_cur in enumerate(driveit_ordering):
            print(f"current {name_cur}")
            if driveit_output_next is None:
                driveit_output_next_concat = None
            else:
                print(f"driveit_output_next_concat {driveit_name_next}")
                assert name_cur == driveit_name_next
                driveit_output_next_concat = torch.cat(driveit_output_next, dim=0)
            
            driveit_input_cur = []
            get_descendant(new_net, name_cur)._driveit_phase = "collect_input"
            get_descendant(new_net, name_cur)._driveit_input = driveit_input_cur

            if ix_cur + 1 < len(driveit_ordering):
                name_next = driveit_ordering[ix_cur + 1]
                driveit_name_next = name_next
                driveit_output_next = []
                get_descendant(new_net, name_next)._driveit_phase = "collect_output"
                get_descendant(new_net, name_next)._driveit_output = driveit_output_next
                print(f"setting {name_next} collect_output")

            acc = 0.0
            for data, label in test_loader:
                # Modifies driveit_input_cur and driveit_output_next
                output = new_net(data)
                acc += (output.argmax(dim=1) == label).float().mean()
            acc = acc / len(test_loader)
            print(f"driveit-bb-pluto before replacing {name_cur}: acc={acc}")

            # Compute hash function and LUT
            print(name_cur)
            driveit_input_cur_concat = torch.cat(driveit_input_cur, dim=0)
            # Assumes the forward-pass in each iteration sees the same data
            get_descendant(new_net, name_cur).fit_lut(
                driveit_input_cur_concat, driveit_output_next_concat)

            # Switch to hash & LUT
            get_descendant(new_net, name_cur)._driveit_phase = "apply_lut"

            # Restore noop phase to nextnext linear layer
            if ix_cur + 2 < len(driveit_ordering):
                name_nextnext = driveit_ordering[ix_cur + 2]
                get_descendant(new_net, name_nextnext)._driveit_phase = "noop"

            # XXX - what about residual connections?

            # cannot do this because appends to driveit_output
            #output = torch.softmax(new_net(image).squeeze(), dim = 0)
            #values, indices = torch.topk(output, k = 5)
            #class_names = [cifar10_classes[i] for i in indices.numpy().astype(int)]
            #values = values.detach().numpy()
            #print(f"after BBPluto {class_names} {values}")
    """
