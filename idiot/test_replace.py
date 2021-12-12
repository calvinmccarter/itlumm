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

    idiot_ordering = []  # ordered list of IdiotLinear layers
    max_collect_samples = 1024
    algorithm = "pluto"
    idiot_opts = {
        "max_collect_samples": max_collect_samples,
        "ncodebooks": None,
        "nonzeros_heuristic": "r2",
        "algorithm": algorithm,
        "objective": "mse-sklearn",
    }

    # XXX - only fc1 in MLPMixer.MLP has gelu
    # f_act = F.gelu
    f_act = None

    new_net = replace_descendants(
        net,
        idiot_ordering,
        idiot_opts,
        "",
        f_act,
    )
    #print(new_net)

    new_net.eval()
    output = torch.softmax(new_net(image).squeeze(), dim = 0)
    values, indices = torch.topk(output, k = 5)
    class_names = [cifar10_classes[i] for i in indices.numpy().astype(int)]
    values = values.detach().numpy()
    print(f"after Linear->IdiotLinear {class_names} {values}")

    set_all_descendant_attrs(new_net, "_idiot_phase", "find_ordering")
    output = new_net(image)  # mutates idiot_ordering
    set_all_descendant_attrs(new_net, "_idiot_phase", "noop")


    def f_softmax(x):
        return torch.softmax(x, dim=1)
    #print(new_net)
    #print(output)
    #print(f_softmax(output))
    #print(get_descendant(new_net, idiot_ordering[-1]))
    get_descendant(
        new_net, idiot_ordering[-1])._idiot_activation = f_softmax
    #print(new_net)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=8,
        shuffle=False,
        num_workers=1,
    )

    # PLUTO
    with torch.no_grad():
        for lname in idiot_ordering:
            idiot_input = []  # list for storing all activations
            get_descendant(new_net, lname)._idiot_phase = "collect_input"
            get_descendant(new_net, lname)._idiot_input = idiot_input
            acc = 0.0
            for data, label in test_loader:
                # Modifies idiot_input_cur and idiot_output_next
                output = new_net(data)
                acc += (output.argmax(dim=1) == label).float().mean()
            acc = acc / len(test_loader)
            print(f"idiot-{algorithm}: before replacing {lname}: acc={acc}")

            idiot_input_concat = torch.cat(idiot_input, dim=0)
            get_descendant(new_net, lname).fit_lut(idiot_input_concat, None)
            get_descendant(new_net, lname)._idiot_phase = "apply_lut"

            output = torch.softmax(new_net(image).squeeze(), dim = 0)
            values, indices = torch.topk(output, k = 5)
            class_names = [cifar10_classes[i] for i in indices.numpy().astype(int)]
            values = values.detach().numpy()
            print(f"after {algorithm} {class_names} {values}")
        acc = 0.0
        for data, label in test_loader:
            # Modifies idiot_input_cur and idiot_output_next
            output = new_net(data)
            acc += (output.argmax(dim=1) == label).float().mean()
        acc = acc / len(test_loader)
        print(f"idiot-{algorithm}: final {max_collect_samples}: acc={acc}")


    # BBPLUTO
    """
    print("\n".join(idiot_ordering))
    with torch.no_grad():
        idiot_output_next = None
        idiot_name_next = None
        for ix_cur, name_cur in enumerate(idiot_ordering):
            print(f"current {name_cur}")
            if idiot_output_next is None:
                idiot_output_next_concat = None
            else:
                print(f"idiot_output_next_concat {idiot_name_next}")
                assert name_cur == idiot_name_next
                idiot_output_next_concat = torch.cat(idiot_output_next, dim=0)
            
            idiot_input_cur = []
            get_descendant(new_net, name_cur)._idiot_phase = "collect_input"
            get_descendant(new_net, name_cur)._idiot_input = idiot_input_cur

            if ix_cur + 1 < len(idiot_ordering):
                name_next = idiot_ordering[ix_cur + 1]
                idiot_name_next = name_next
                idiot_output_next = []
                get_descendant(new_net, name_next)._idiot_phase = "collect_output"
                get_descendant(new_net, name_next)._idiot_output = idiot_output_next
                print(f"setting {name_next} collect_output")

            acc = 0.0
            for data, label in test_loader:
                # Modifies idiot_input_cur and idiot_output_next
                output = new_net(data)
                acc += (output.argmax(dim=1) == label).float().mean()
            acc = acc / len(test_loader)
            print(f"idiot-bb-pluto before replacing {name_cur}: acc={acc}")

            # Compute hash function and LUT
            print(name_cur)
            idiot_input_cur_concat = torch.cat(idiot_input_cur, dim=0)
            # Assumes the forward-pass in each iteration sees the same data
            get_descendant(new_net, name_cur).fit_lut(
                idiot_input_cur_concat, idiot_output_next_concat)

            # Switch to hash & LUT
            get_descendant(new_net, name_cur)._idiot_phase = "apply_lut"

            # Restore noop phase to nextnext linear layer
            if ix_cur + 2 < len(idiot_ordering):
                name_nextnext = idiot_ordering[ix_cur + 2]
                get_descendant(new_net, name_nextnext)._idiot_phase = "noop"

            # XXX - what about residual connections?

            output = torch.softmax(new_net(image).squeeze(), dim = 0)
            values, indices = torch.topk(output, k = 5)
            class_names = [cifar10_classes[i] for i in indices.numpy().astype(int)]
            values = values.detach().numpy()
            print(f"after BBPluto {class_names} {values}")
    """
