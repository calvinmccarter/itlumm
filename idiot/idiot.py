from collections import OrderedDict
from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

import bolt.experiments.python.vq_amm as vq_amm

DEFAULT_IDIOT_OPTS = {
    "max_input_len": 1e6,
    "max_input_numel": 1e8,
    "ncodebooks": None,
}

class IdiotLinear(nn.Linear):
    r"""Linear that supports ITLUMM.

    Args:
        activation: nonlinear function that comes after this layer.
            Default: ``None``
        ordering: OrderedDict used to find the order of Linears.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        idiot_phase=None,
        idiot_ordering=None,
        idiot_input=None,
        idiot_opts=None,
        idiot_name=None,
        idiot_activation=None,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self._idiot_phase = idiot_phase  # external
        self._idiot_ordering = idiot_ordering  # external
        self._idiot_input = idiot_input  # external
        self._idiot_opts = deepcopy(DEFAULT_IDIOT_OPTS)
        self._idiot_opts.update(deepcopy(idiot_opts) or {})
        self._idiot_name = deepcopy(idiot_name)
        self._idiot_activation = deepcopy(idiot_activation)

        idiot_phase[idiot_name] = "noop"

        self._pluto = None

    def forward(self, input: Tensor) -> Tensor:
        phase = self._idiot_phase[self._idiot_name]
        if phase == "find_ordering":
            self._idiot_ordering.append(self._idiot_name)
        elif phase == "collect_input":
            cur_len = input.shape[0] + sum(
                [inp.shape[0] for inp in self._idiot_input]
            )
            cur_numel = math.prod(input.shape) + sum(
                [math.prod(inp.shape) for inp in self._idiot_input]
            )
            max_len = self._idiot_opts["max_input_len"]
            max_numel = self._idiot_opts["max_input_numel"]
            if cur_len <= max_len and cur_numel <= max_numel:
                self._idiot_input.append(input)
        elif phase == "apply_lut":
            #rint(f"apply_lut {self._idiot_name}")
            #rint(f"input shape {input.shape}")
            #rint(f"weight shape {self.weight.data.shape}")
            input_shape = input.shape
            input_np = input.reshape((-1, input.shape[-1])).numpy()
            #rint(f"input_np.shape: {input_np.shape}")
            self._pluto.reset_for_new_task()
            output_np = self._pluto(
                input_np,
                self.weight.data.transpose(0, 1).numpy(),
                bias=self.bias.data.numpy(),
            )
            #rint(f"output_np.shape: {output_np.shape}")
            output_shape = tuple(list(input_shape[:-1]) + [output_np.shape[-1]])
            #rint(f"output_shape: {output_shape}")
            output = torch.from_numpy(output_np).reshape(output_shape)
            output += self.bias.data
            return output
        
        return F.linear(input, self.weight, self.bias)

    def fit_lut(self):
        ncodebooks = self._idiot_opts["ncodebooks"]
        (out_features, in_features) = self.weight.data.shape
        if ncodebooks is None:
            ncodebooks = min(in_features // 4, out_features)
        print(f"fitting pluto (in, out) = {(in_features, out_features)} with {ncodebooks} ncodebooks")
        self._pluto = vq_amm.PlutoMatmul(ncodebooks=ncodebooks)
        # TODO- minimize n_codebooks s.t. 
        #   n_codebooks < self.weight.shape[0]
        #   accuracy loss < 0.99^(1/num_linears)
        #   n_ops_pluto < n_ops_original
        #   or, keep increasing ncodebooks until good enough
        input = torch.cat(self._idiot_input, dim=0)
        input_np = input.reshape((-1, input.shape[-1])).numpy()
        #rint(f"fit_lut {self._idiot_name}")
        #rint(self)
        #rint(input_np.shape)
        #rint(self.weight.shape)
        #rint(self.bias.shape)
        self._pluto.fit(
            input_np,
            self.weight.data.transpose(0, 1).numpy(),
            bias=self.bias.data.numpy(),
        )

    def extra_repr(self) -> str:
        super_repr = [super().extra_repr(),]
        sin = self._idiot_name
        idx = self._idiot_ordering.index(sin)
        my_repr = [
            f"_idiot_phase[self._idiot_name]={self._idiot_phase[sin]}",
            f"_idiot_ordering.index(\"{sin}\")={idx}",
            f"_idiot_opts={self._idiot_opts}",
            f"_idiot_name={self._idiot_name}",
            f"_idiot_activation={self._idiot_activation}",
        ]
        return ", ".join(super_repr + my_repr)


def replace_linear(
    mod: nn.Linear,
    idiot_phase,
    idiot_ordering,
    idiot_input,
    idiot_opts,
    idiot_name,
    idiot_activation,
):
    """
    newmod = IdiotLinear(
        in_features=mod.in_features,
        out_features=mod.out_features,
        bias=mod.bias is not None,
        device=mod.device,
        dtype=mod.dtype,
    )
    newmod.weight = mod.weight
    newmod.bias = mod.bias
    if mod.training:
        newmod.train()
    else:
        newmod.eval()
    """
    newmod = IdiotLinear(
        in_features=mod.in_features,
        out_features=mod.out_features,
        idiot_phase=idiot_phase,
        idiot_ordering=idiot_ordering,
        idiot_input=idiot_input,
        idiot_opts=idiot_opts,
        idiot_name=idiot_name,
        idiot_activation=idiot_activation,
    )
    newmod.load_state_dict(
        deepcopy(mod.state_dict()), strict=False)

    return newmod


def replace_descendants(
    mod: nn.Module,
    idiot_phase,
    idiot_ordering,
    idiot_input,
    idiot_opts,
    idiot_name,
    idiot_activation,
):
    if type(mod) == nn.Linear:
        new_mod = replace_linear(
            mod,
            idiot_phase=idiot_phase,
            idiot_ordering=idiot_ordering,
            idiot_input=idiot_input,
            idiot_opts=idiot_opts,
            idiot_name=idiot_name,
            idiot_activation=idiot_activation,
        )
        return new_mod

    new_children = {}
    for name, child in mod.named_children():
        fullname = idiot_name + "." + name
        new_children[name] = replace_descendants(
            child,
            idiot_phase=idiot_phase,
            idiot_ordering=idiot_ordering,
            idiot_input=idiot_input,
            idiot_opts=idiot_opts,
            idiot_name=fullname,
            idiot_activation=idiot_activation,
        )
    for name, child in new_children.items():
        mod._modules[name] = child
    return mod


def get_descendant_by_fullname(
    mod,
    fullname,
):
    """

    Returns:
        reference to Module if found, None otherwise.
    """
    if type(mod) == IdiotLinear and mod._idiot_name == fullname:
        return mod
    for name, child in mod.named_children():
        maybe = get_descendant_by_fullname(child, fullname)
        if maybe is not None:
            return maybe
    return None