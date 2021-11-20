import torch
import torch.multiprocessing as tmp
import torchvision as tv
import torch.nn.functional as F
import torchvision.transforms as tvt

from PIL import Image
from torchvision import transforms

import idiot

from mlp_mixer import MLPMixer

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
net.load_state_dict(torch.load("mlp_mixer_cifar10_small.pt"))
#print(net)

transform = tvt.Compose([
    tvt.ToTensor(),
    tvt.Resize((32, 32)),
    tvt.RandomHorizontalFlip(),
    tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) 

idiot_phase = {}  # mapping layer fullname to phase str
idiot_ordering = []  # ordered list of IdiotLinear layers
idiot_input = []  # list for storing all activations
idiot_opts = {
    "max_input_len": 1000,
    "max_input_numel": 32*64*10,
}

new_net = idiot.replace_descendents(
    net, idiot_phase, idiot_ordering, idiot_input, idiot_opts, "", F.gelu)
#print(new_net)

filename = 'images/frog.jpg'
image = Image.open(filename)
image = transform(image)
image = image.unsqueeze(0)

for lname in idiot_phase:
    idiot_phase[lname] = "find_ordering"
output = new_net(image)  # mutates ordering
for lname in idiot_phase:
    idiot_phase[lname] = "noop"


def f_softmax(x):
    return torch.softmax(x, dim=1)
print(new_net)
#print(output)
#print(f_softmax(output))
#print(idiot.get_descendant_by_fullname(new_net, idiot_ordering[-1]))
idiot.get_descendant_by_fullname(
    new_net, idiot_ordering[-1])._idiot_activation = f_softmax
print(new_net)

if __name__ == "__main__" and False:
    tmp.freeze_support()
    test_data = tv.datasets.CIFAR10(
        './',
        train=False,
        transform=transform,
        target_transform=None,
        download=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=32,
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
            print(idiot_input_concat.shape)

            idiot.get_descendant_by_fullname(new_net, lname).fit_lut()
            idiot_phase[lname] = "apply_lut"

            # TODO fine-tune
