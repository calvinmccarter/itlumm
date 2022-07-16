from collections import OrderedDict
from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

import bolt.experiments.python.vq_amm as vq_amm


DEFAULT_IDIOT_OPTS = {
    "algorithm": "pluto",
    "max_collect_samples": 1e6,
    "ncodebooks": None,
    "nonzeros_heuristic": "pq",
    "objective": "mse",
    "accumulate_how": "mean",
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
        driveit_ordering=None,
        driveit_opts=None,
        driveit_name=None,
        driveit_activation=None,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

        self._driveit_ordering = driveit_ordering  # external
        self._driveit_opts = deepcopy(DEFAULT_IDIOT_OPTS)
        self._driveit_opts.update(deepcopy(driveit_opts) or {})
        self._driveit_name = deepcopy(driveit_name)
        self._driveit_activation = deepcopy(driveit_activation)

        self._driveit_phase = "noop"
        self._driveit_input = None
        self._driveit_output = None
        self._lut = None

    def forward(self, input: Tensor) -> Tensor:
        if self._driveit_phase == "apply_lut":
            #rint(f"apply_lut {self._driveit_name}")
            #rint(f"input shape {input.shape}")
            #rint(f"weight shape {self.weight.data.shape}")
            input_shape = input.shape
            input_np = input.reshape((-1, input.shape[-1])).detach().numpy()
            #rint(f"input_np.shape: {input_np.shape}")
            self._lut.reset_for_new_task()
            output_np = self._lut(
                input_np,
                self.weight.data.transpose(0, 1).numpy(),
            )
            #rint(f"output_np.shape: {output_np.shape}")
            output_shape = tuple(list(input_shape[:-1]) + [output_np.shape[-1]])
            #rint(f"output_shape: {output_shape}")
            output = torch.from_numpy(output_np).reshape(output_shape)
            output += self.bias.data
            return output
        elif self._driveit_phase == "find_ordering":
            self._driveit_ordering.append(self._driveit_name)
            return F.linear(input, self.weight, self.bias)
        elif self._driveit_phase == "collect_input":
            if self._driveit_input is None:
                raise ValueError("needs _driveit_input for collect_input phase")
            cur_len = input.shape[0] + sum(
                [inp.shape[0] for inp in self._driveit_input]
            )
            max_len = self._driveit_opts["max_collect_samples"]
            if cur_len <= max_len:
                self._driveit_input.append(input)
            #rint(f"collect_input cur_len:{cur_len} {self._driveit_name}")
            return F.linear(input, self.weight, self.bias)
        elif self._driveit_phase == "collect_output":
            # only used in BBPLUTO
            if self._driveit_output is None:
                raise ValueError("needs _driveit_output for collect_output")
            output = F.linear(input, self.weight, self.bias)
            cur_len = output.shape[0] + sum(
                [outp.shape[0] for outp in self._driveit_output]
            )
            max_len = self._driveit_opts["max_collect_samples"]
            if cur_len <= max_len:
                self._driveit_output.append(output - self.bias)
            #rint(f"collect_output cur_len:{cur_len} {self._driveit_name}")
            return output
        elif self._driveit_phase == "noop":
            return F.linear(input, self.weight, self.bias)
        else:
            raise ValueError(f"unexpected _driveit_phase: {self._driveit_phase}")
            

    def fit_lut(self, input, output):
        """

        Args:
            input: A from A @ B
            output: desired A @ B - not including bias or activation
        """
        ncodebooks = self._driveit_opts["ncodebooks"]
        algorithm = self._driveit_opts["algorithm"]
        (n_out, n_in) = self.weight.data.shape
        if ncodebooks is None:
            ncodebook_factor = 2
            ncodebooks = 2 ** math.floor(math.log2(n_in // ncodebook_factor))
            # XXX upcast_every assertion
            ncodebooks = min(ncodebooks, 256)
        elif ncodebooks <= -1:
            ncodebook_factor = abs(ncodebooks)
            ncodebooks = 2 ** math.floor(math.log2(n_in // ncodebook_factor))
            # XXX upcast_every assertion
            ncodebooks = min(ncodebooks, 256)

        self._driveit_opts["actual_ncodebooks"] = ncodebooks

        print(f"fit_lut {algorithm} (in, out)={(n_in, n_out)}=>{ncodebooks}")

        # Reshape input and output to be 2d matrices, not tensors
        input_np = input.reshape((-1, input.shape[-1])).detach().numpy()
        #rint(f"fit_lut {self._driveit_name}")
        #rint(self)
        #rint(f"input: {input.shape} -> {input_np.shape}")
        if output is not None:
            print(f"output: {output.shape if output is not None else None}")
        #rint(f"weight: {self.weight.shape}   bias:{self.bias.shape}")
        if output is None:
            output_np = None
        else:
            assert output.shape == tuple(list(input.shape[:-1]) + [n_out])
            output_np = output.reshape((-1, output.shape[-1])).numpy()
        # TODO- minimize n_codebooks s.t. 
        #   n_codebooks < self.weight.shape[0]
        #   accuracy loss < 0.99^(1/num_linears)
        #   n_ops_pluto < n_ops_original
        #   or, keep increasing ncodebooks until good enough

        if algorithm == "pluto":
            self._lut = vq_amm.PlutoMatmul(
                ncodebooks=ncodebooks,
                activation=self._driveit_activation,
                nonzeros_heuristic=self._driveit_opts["nonzeros_heuristic"],
                objective=self._driveit_opts["objective"],
                accumulate_how=self._driveit_opts["accumulate_how"],
            )
            self._lut.fit(
                input_np,
                self.weight.data.transpose(0, 1).numpy(),
                output=output_np,
                bias=self.bias.data.numpy(),
            )
        elif algorithm == "mithral":
            self._lut = vq_amm.MithralMatmul(
                ncodebooks=ncodebooks,
                nonzeros_heuristic=self._driveit_opts["nonzeros_heuristic"],
            )
            self._lut.fit(
                input_np,
                self.weight.data.transpose(0, 1).numpy(),
            )
        else:
            raise ValueError("invalid algorithm: {algorithm}")


    def extra_repr(self) -> str:
        super_repr = [super().extra_repr(),]
        sin = self._driveit_name
        idx = self._driveit_ordering.index(sin)
        my_repr = [
            f"_driveit_ordering.index(\"{sin}\")={idx}",
            f"_driveit_phase={self._driveit_phase}",
            f"_driveit_opts={self._driveit_opts}",
            f"_driveit_name={self._driveit_name}",
            f"_driveit_activation={self._driveit_activation}",
        ]
        return ", ".join(super_repr + my_repr)


def replace_linear(
    mod: nn.Linear,
    driveit_ordering,
    driveit_opts,
    driveit_name,
    driveit_activation,
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
        driveit_ordering=driveit_ordering,
        driveit_opts=driveit_opts,
        driveit_name=driveit_name,
        driveit_activation=driveit_activation,
    )
    newmod.load_state_dict(
        deepcopy(mod.state_dict()), strict=False)

    return newmod


def replace_descendants(
    mod: nn.Module,
    driveit_ordering,
    driveit_opts,
    driveit_name,
    driveit_activation,
):
    if type(mod) == nn.Linear:
        new_mod = replace_linear(
            mod,
            driveit_ordering=driveit_ordering,
            driveit_opts=driveit_opts,
            driveit_name=driveit_name,
            driveit_activation=driveit_activation,
        )
        return new_mod

    new_children = {}
    for name, child in mod.named_children():
        fullname = driveit_name + "." + name
        new_children[name] = replace_descendants(
            child,
            driveit_ordering=driveit_ordering,
            driveit_opts=driveit_opts,
            driveit_name=fullname,
            driveit_activation=driveit_activation,
        )
    for name, child in new_children.items():
        mod._modules[name] = child
    return mod


def set_all_descendant_attrs(
    mod,
    name,
    value,
):
    r"""Sets all IdiotLinear descendants' attrname to attrvalue.

    Args:
        mod: nn.Module
            Top-level network that contains all IdiotLinear modules.
        name: str
            Name of attribute we want to modify.
        value: str
            Value we want to apply to the attribute. Value is deepcopied.
    """
    if type(mod) == IdiotLinear:
        assert hasattr(mod, name)
        setattr(mod, name, deepcopy(value))
    for _, child in mod.named_children():
        set_all_descendant_attrs(child, name, value)


def get_descendant(
    mod,
    fullname,
):
    r"""

    Args:

        mod: nn.Module
            Top-level network that contains the desired IdiotLinear module.
        fullname: str
            Search str applied to IdiotLinear module _driveit_name attribute.

    Returns:
        reference to Module if found, None otherwise.
    """
    if type(mod) == IdiotLinear and mod._driveit_name == fullname:
        return mod
    for name, child in mod.named_children():
        maybe = get_descendant(child, fullname)
        if maybe is not None:
            return maybe
    return None
