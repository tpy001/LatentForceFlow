import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        loaded_params = _augment_with_moe_shared_ffn_weights(loaded_params, params)
        loaded_params = _augment_with_mor_action_expert_weights(loaded_params, params)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _augment_with_moe_shared_ffn_weights(loaded_params: at.Params, params: at.Params) -> at.Params:
    """Copies base FFN weights into MOE shared expert keys when shapes are compatible.

    Mapping examples:
      .../mlp/...   -> .../moe/expert_0/...
      .../mlp_1/... -> .../moe_1/expert_0/...
    """
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    augmented = dict(flat_loaded)

    # Capture ".../mlp<suffix>/<rest>", suffix can be "" or "_1", "_2", etc.
    mlp_pattern = re.compile(r"^(.*?/)(mlp(?:_\d+)?)/(.*)$")

    copied = 0
    for k, v in flat_loaded.items():
        m = mlp_pattern.match(k)
        if m is None:
            continue
        prefix, mlp_name, rest = m.groups()
        moe_name = mlp_name.replace("mlp", "moe", 1)
        mapped_key = f"{prefix}{moe_name}/expert_0/{rest}"
        if mapped_key in augmented:
            continue
        if mapped_key not in flat_ref:
            continue
        if hasattr(v, "shape") and hasattr(flat_ref[mapped_key], "shape") and v.shape != flat_ref[mapped_key].shape:
            continue
        augmented[mapped_key] = v
        copied += 1

    if copied > 0:
        logger.info("Mapped %d FFN tensors from base MLP to MOE shared expert.", copied)

    return flax.traverse_util.unflatten_dict(augmented, sep="/")


def _augment_with_mor_action_expert_weights(loaded_params: at.Params, params: at.Params) -> at.Params:
    """Copies action-expert (_1) checkpoint tensors into MOR refine-expert (_2) slots.

    This is useful when loading a 2-expert pi0 checkpoint into a 3-expert MOR model:
      .../attn_1/...       -> .../attn_2/...
      .../mlp_1/...        -> .../mlp_2/...
      .../final_norm_1/... -> .../final_norm_2/...
    """
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    augmented = dict(flat_loaded)

    copied = 0
    for k, v in flat_loaded.items():
        parts = k.split("/")
        mapped_parts = [part[:-2] + "_2" if part.endswith("_1") else part for part in parts]
        mapped_key = "/".join(mapped_parts)
        if mapped_key == k:
            continue
        if mapped_key in augmented:
            continue
        if mapped_key not in flat_ref:
            continue
        if hasattr(v, "shape") and hasattr(flat_ref[mapped_key], "shape") and v.shape != flat_ref[mapped_key].shape:
            continue
        augmented[mapped_key] = v.astype(flat_ref[mapped_key].dtype) if v.dtype != flat_ref[mapped_key].dtype else v
        copied += 1

    if copied > 0:
        logger.info("Mapped %d tensors from action expert (_1) to MOR refine expert (_2).", copied)

    return flax.traverse_util.unflatten_dict(augmented, sep="/")
