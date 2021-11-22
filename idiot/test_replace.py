import pickle
import torch
import torch.multiprocessing as tmp
import torchvision as tv
import torch.nn.functional as F
import torchvision.transforms as tvt

from PIL import Image
from torchvision import transforms

from idiot.idiot import (
    get_descendant_by_fullname,
    replace_descendants,
)

from idiot.mlp_mixer import MLPMixer


if __name__ == "__main__":
    with open("/Users/calvinm/sandbox/objective-correlative/cifar-10-batches-py/batches.meta", 'rb') as pickleFile:
        cifar10_classes = pickle.load(pickleFile)['label_names'] 
    transform = tvt.Compose([
        tvt.ToTensor(),
        tvt.Resize((32, 32)),
        #tvt.RandomHorizontalFlip(),
        tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
    filename = '/Users/calvinm/sandbox/MLP-Mixer/images/frog.jpg'
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
        "/Users/calvinm/sandbox/MLP-Mixer/mlp_mixer_cifar10_small.pt"))
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
        "/Users/calvinm/sandbox/MLP-Mixer/training_artefacts/model_epoch_41.pth"))

    net.eval()
    output = torch.softmax(net(image).squeeze(), dim = 0)
    values, indices = torch.topk(output, k = 5)
    class_names = [cifar10_classes[i] for i in indices.numpy().astype(int)]
    values = values.detach().numpy()
    print(f"original MLPMixer {class_names} {values}")

    idiot_phase = {}  # mapping layer fullname to phase str
    idiot_ordering = []  # ordered list of IdiotLinear layers
    idiot_input = []  # list for storing all activations
    idiot_opts = {
        "max_input_len": 1000,
        "max_input_numel": 1e6, #32*64*10,
        "ncodebooks": 16,
    }

    # XXX - only fc1 in MLPMixer.MLP has gelu
    # f_act = F.gelu
    f_act = None

    new_net = replace_descendants(
        net,
        idiot_phase,
        idiot_ordering,
        idiot_input,
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

    for lname in idiot_phase:
        idiot_phase[lname] = "find_ordering"
    output = new_net(image)  # mutates ordering
    for lname in idiot_phase:
        idiot_phase[lname] = "noop"


    def f_softmax(x):
        return torch.softmax(x, dim=1)
    #print(new_net)
    #print(output)
    #print(f_softmax(output))
    #print(get_descendant_by_fullname(new_net, idiot_ordering[-1]))
    get_descendant_by_fullname(
        new_net, idiot_ordering[-1])._idiot_activation = f_softmax
    #print(new_net)

    test_data = tv.datasets.CIFAR10(
        './',
        train=False,
        transform=transform,
        target_transform=None,
        download=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=4,
        shuffle=False,
        num_workers=1,
    )

    with torch.no_grad():
        for lname in idiot_ordering:
            idiot_phase[lname] = "collect_input"
            idiot_input.clear()
            for data, label in test_loader:
                output = new_net(data)
            #idiot_phase[lname] = "noop"
            idiot_input_concat = torch.cat(idiot_input, dim=0)
            print(lname)

            get_descendant_by_fullname(new_net, lname).fit_lut()
            idiot_phase[lname] = "apply_lut"

            output = torch.softmax(new_net(image).squeeze(), dim = 0)
            values, indices = torch.topk(output, k = 5)
            class_names = [cifar10_classes[i] for i in indices.numpy().astype(int)]
            values = values.detach().numpy()
            print(f"after Pluto {class_names} {values}")


            # TODO fine-tune
