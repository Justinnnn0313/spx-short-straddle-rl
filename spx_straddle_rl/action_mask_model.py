from __future__ import annotations

from gymnasium.spaces import Dict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch()


class TorchActionMaskModel(TorchModelV2, nn.Module):
    """Discrete-action masking model for RLlib's old Torch ModelV2 API."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        if not (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        ):
            raise ValueError(
                "TorchActionMaskModel requires a Dict observation with "
                "'observations' and 'action_mask' keys."
            )

        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"].float()
        no_valid = action_mask.sum(dim=1, keepdim=True) <= 0.0
        action_mask = torch.where(no_valid, torch.ones_like(action_mask), action_mask)
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})
        inf_mask = torch.where(
            action_mask > 0.0,
            torch.zeros_like(action_mask),
            torch.full_like(action_mask, -1.0e9),
        )
        return logits + inf_mask, state

    def value_function(self):
        return self.internal_model.value_function()
