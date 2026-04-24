from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import torch
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize
from collections.abc import Sequence
import dataclasses
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Protocol, TypeAlias, runtime_checkable
from typing import Union

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class AddEffortTokenizerField(DataTransformFn):
    """Copies raw effort into a dedicated field for tokenizer usage before normalization."""

    source_keys: Sequence[str] = ("effort", "observation.effort")
    target_key: str = "effort_tokenizer"

    def __call__(self, data: DataDict) -> DataDict:
        if self.target_key in data:
            return data

        for source_key in self.source_keys:
            if source_key in data and data[source_key] is not None:
                return {**data, self.target_key: data[source_key]}
        return data


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False


    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None
        
     
        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )

@dataclasses.dataclass(frozen=True)
class QuaternionToEuler(DataTransformFn):
    """将四元数(wxyz格式)转换为欧拉角(roll, pitch, yaw)。
    
    四元数格式: [w, x, y, z]
    欧拉角格式: [roll, pitch, yaw] (单位:弧度)
    使用ZYX顺序(先绕Z轴yaw,再绕Y轴pitch,最后绕X轴roll)
    """
    quat_key: str = "observation.state"
    euler_key: str = "observation.state"
    euler_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # 哪个轴在180度附近就改哪个,防止欧拉角周期性跳变
    
    def __call__(self, data: dict) -> dict:
        if self.quat_key not in data:
            return data
        
        state = data[self.quat_key]
        is_torch = isinstance(state, torch.Tensor)
        
        # 提取各部分
        pos = state[..., :3]  # 位置 [x, y, z]
        quat = state[..., 3:7]  # 四元数 [w, x, y, z]
        
        # 处理gripper - 根据类型选择空数组
        if state.shape[-1] > 7:
            gripper = state[..., 7:]
        else:
            gripper = torch.tensor([]) if is_torch else np.array([])
        
        # 四元数转欧拉角
        euler = self._quat_to_euler(quat)
        
        # 拼接状态 - 根据类型选择拼接方法
        if is_torch:
            new_state_parts = [pos, euler]
            if gripper.numel() > 0:
                new_state_parts.append(gripper)
            new_state = torch.cat(new_state_parts, dim=-1)
        else:
            new_state_parts = [pos, euler]
            if gripper.size > 0:
                new_state_parts.append(gripper)
            new_state = np.concatenate(new_state_parts, axis=-1)
        
        data[self.euler_key] = new_state
        return data
    
    def _quat_to_euler(self, quat: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """将四元数转换为欧拉角。
        
        Args:
            quat: 形状为 (..., 4) 的tensor或array,最后一维为 [w, x, y, z]
            
        Returns:
            形状为 (..., 3) 的tensor或array,最后一维为 [roll, pitch, yaw]
        """
        is_torch = isinstance(quat, torch.Tensor)
        
        # 保存原始形状和类型信息
        original_shape = quat.shape
        batch_shape = original_shape[:-1]
        
        # 转换为numpy进行计算
        if is_torch:
            quat_np = quat.detach().cpu().numpy().reshape(-1, 4)
            dtype = quat.dtype
            device = quat.device
        else:
            quat_np = quat.reshape(-1, 4)
            dtype = quat.dtype
        
        # scipy格式转换: [w,x,y,z] -> [x,y,z,w]
        quat_scipy = quat_np[:, [1, 2, 3, 0]]
        
        # 使用scipy转换
        r = Rotation.from_quat(quat_scipy)
        euler_np = r.as_euler('xyz', degrees=False)  # [roll, pitch, yaw]
        
        # 恢复原始形状
        euler_np = euler_np.reshape(batch_shape + (3,))
        euler_np = euler_np + np.array(self.euler_offset)
        euler_np = (euler_np + np.pi) % (2 * np.pi) - np.pi
        
        # 转换回原始类型
        if is_torch:
            euler = torch.tensor(euler_np, dtype=dtype, device=device)
        else:
            euler = euler_np
        
        return euler


@dataclasses.dataclass(frozen=True)
class EulerToQuaternion(DataTransformFn):
    """将欧拉角(roll, pitch, yaw)转换为四元数(wxyz格式)。
    
    欧拉角格式: [roll, pitch, yaw] (单位:弧度)
    四元数格式: [w, x, y, z]
    使用ZYX顺序(先绕Z轴yaw,再绕Y轴pitch,最后绕X轴roll)
    """
    euler_key: str = "observation.state"
    quat_key: str = "observation.state"
    euler_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # 必须与上面保持一致
    
    def __call__(self, data: dict) -> dict:
        if self.euler_key not in data:
            return data
        
        state = data[self.euler_key]
        is_torch = isinstance(state, torch.Tensor)
        
        # 提取各部分
        pos = state[..., :3]  # 位置 [x, y, z]
        euler = state[..., 3:6]  # 欧拉角 [roll, pitch, yaw]
        
        # 处理gripper - 根据类型选择空数组
        if state.shape[-1] > 6:
            gripper = state[..., 6:]
        else:
            gripper = torch.tensor([]) if is_torch else np.array([])
        
        # 欧拉角转四元数
        quat = self._euler_to_quat(euler)
        
        # 拼接状态 - 根据类型选择拼接方法
        if is_torch:
            new_state_parts = [pos, quat]
            if gripper.numel() > 0:
                new_state_parts.append(gripper)
            new_state = torch.cat(new_state_parts, dim=-1)
        else:
            new_state_parts = [pos, quat]
            if gripper.size > 0:
                new_state_parts.append(gripper)
            new_state = np.concatenate(new_state_parts, axis=-1)
        
        data[self.quat_key] = new_state
        return data
    
    def _euler_to_quat(self, euler: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """将欧拉角转换为四元数。
        
        Args:
            euler: 形状为 (..., 3) 的tensor或array,最后一维为 [roll, pitch, yaw]
            
        Returns:
            形状为 (..., 4) 的tensor或array,最后一维为 [w, x, y, z]
        """
        is_torch = isinstance(euler, torch.Tensor)
        
        # 保存原始形状和类型信息
        original_shape = euler.shape
        batch_shape = original_shape[:-1]
        
        # 转换为numpy进行计算
        if is_torch:
            euler_np = euler.detach().cpu().numpy().reshape(-1, 3)
            dtype = euler.dtype
            device = euler.device
        else:
            euler_np = euler.reshape(-1, 3)
            dtype = euler.dtype
        
        euler_np = euler_np + np.array(self.euler_offset)
        euler_np = (euler_np + np.pi) % (2 * np.pi) - np.pi
        
        # 使用scipy转换
        r = Rotation.from_euler('xyz', euler_np, degrees=False)
        quat_scipy = r.as_quat()  # [x, y, z, w]
        
        # scipy格式转换: [x,y,z,w] -> [w,x,y,z]
        quat_np = quat_scipy[:, [3, 0, 1, 2]]
        
        # 恢复原始形状
        quat_np = quat_np.reshape(batch_shape + (4,))
        
        # 转换回原始类型
        if is_torch:
            quat = torch.tensor(quat_np, dtype=dtype, device=device)
            quat = torch.where(quat[..., :1] < 0, -quat, quat)  # w分量在第0位
        else:
            quat = quat_np
            quat = np.where(quat[..., :1] < 0, -quat, quat)

        return quat
