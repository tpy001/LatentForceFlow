"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro
import math
import openpi.policies.libero_policy as libero_policy

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.yuanluo_policy as yuanluo_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.shared.effort_type import EffortType
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None
    max_episodes: int | None = None  # None = 用全部
    rcs_sample_enable: bool = False

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)
    effort_history: Sequence[int] = (0,)


    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)



@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoFlowDepthDataConfig(LeRobotLiberoDataConfig):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/flow_image": "flow_image",
                        "observation/flow_wrist_image": "flow_wrist_image",
                        "observation/depth_image": "depth_image",
                        "observation/depth_wrist_image": "depth_wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoFlowDepthInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )



@dataclasses.dataclass(frozen=True)
class LeRobotYuanluoDataConfig(DataConfigFactory):
    """
    Data config for the custom Yuanluo dataset.
    """

    # Actions are absolute values, so no extra delta transform is needed.
    extra_delta_transform: bool = False
    delta_action_mask_size: int = 6
    delta_action_mask_offset: int = -1
    max_episodes: int | None = None  # None = 用全部
    rcs_sample_enable: bool = False
    

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        ##lerobot
                        "observation.images.head_camera": "observation.images.head_camera",
                        "observation.images.wrist_left_camera": "observation.images.wrist_left_camera",
                        # "observation.images.gelsight_left": "observation.images.gelsight_left",
                        # "observation/gelsight_right": "leftarm_gelsight_right",
                        # "observation/left_wrench": "left_wrench",
                        "observation.state": "observation.state",
                        "action": "action",
                        "prompt": "task",
                        
                        ##tpy dataset
                        # "observation/front_camera": "observation.images.front",
                        # "observation/left_wrist_camera": "observation.images.left_wrist",
                        # "observation/gelsight_left": "observation.images.gelsight_left",
                        # # "observation/gelsight_right": "leftarm_gelsight_right",
                        # # "observation/left_wrench": "left_wrench",
                        # "observation/left_state": "observation.state",
                        # "actions": "action",
                        # # "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[yuanluo_policy.YuanluoInputs(model_type=model_config.model_type,state_dim=7)],
            outputs=[yuanluo_policy.YuanluoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(self.delta_action_mask_size, self.delta_action_mask_offset)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            max_episodes = self.max_episodes
        )

@dataclasses.dataclass(frozen=True)
class LeRobotTaVLADataConfig(DataConfigFactory):
    """
    Data config for the custom Yuanluo dataset.
    """
    max_episodes: int | None = None  # None = 用全部
    rcs_sample_enable: bool = False
    use_delta_joint_actions: bool = True # 是否使用相对于当前frame 的action
    delta_action_mask_size: int = 6
    delta_action_mask_offset: int = -1

    default_prompt: str | None = None
    padding_stat: bool = False
    # Actions are absolute values, so no extra delta transform is needed.
    extra_delta_transform: bool = False
    effort_history: Sequence[int] = ()
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(default=_transforms.Group())
    action_sequence_keys: Sequence[str] = ("action",)
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.images.head_camera": "observation.images.head_camera",
                        "observation.images.wrist_left_camera": "observation.images.wrist_left_camera",
                        # "observation.images.gelsight_left": "observation.images.gelsight_left",
                        "observation.state": "observation.state",
                        "observation.effort": "observation.effort",
                        "observation.is_contact": "observation.is_contact",
                        "action": "action",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                yuanluo_policy.YuanluoTaVLAInputs(
                    model_type=model_config.model_type,
                    use_future_rgb_instead_of_flow=getattr(model_config, "use_future_rgb_instead_of_flow", False),
                )
            ], # here is the difference to parent class
            outputs=[yuanluo_policy.YuanluoTaVLAOutputs()],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(self.delta_action_mask_size, self.delta_action_mask_offset)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        if self.default_prompt and isinstance(self.repo_id, list):
            raise ValueError("Using default prompt when using multiple dataset is incorrect.")
        
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            effort_history=self.effort_history,
            prompt_from_task=(self.default_prompt is None),
            max_episodes = self.max_episodes,
            rcs_sample_enable = self.rcs_sample_enable
        )



@dataclasses.dataclass(frozen=True)
class LeRobotOptimalFlowDataConfig(DataConfigFactory):
    """
    Data config for the custom Yuanluo dataset.
    """
    max_episodes: int | None = None  # None = 用全部
    rcs_sample_enable: bool = False
    use_delta_joint_actions: bool = True # 是否使用相对于当前frame 的action
    delta_action_mask_size: int = 6
    delta_action_mask_offset: int = -1

    default_prompt: str | None = None
    padding_stat: bool = False
    # Actions are absolute values, so no extra delta transform is needed.
    extra_delta_transform: bool = False
    effort_history: Sequence[int] = ()
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(default=_transforms.Group())
    action_sequence_keys: Sequence[str] = ("action",)
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.images.head_camera": "observation.images.head_camera",
                        "observation.images.wrist_left_camera": "observation.images.wrist_left_camera",
                        "observation.images.future_flow": "observation.future_flow.base_0_rgb",
                        "observation.images.future_wrist_flow": "observation.future_flow.left_wrist_0_rgb",
                        "observation.state": "observation.state",
                        "observation.effort": "observation.effort",
                        "observation.is_contact": "observation.is_contact",
                        "action": "action",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                yuanluo_policy.YuanluoTaVLAInputs(
                    model_type=model_config.model_type,
                    use_future_rgb_instead_of_flow=getattr(model_config, "use_future_rgb_instead_of_flow", False),
                )
            ], # here is the difference to parent class
            outputs=[yuanluo_policy.YuanluoTaVLAOutputs()],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(self.delta_action_mask_size, self.delta_action_mask_offset)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        if self.default_prompt and isinstance(self.repo_id, list):
            raise ValueError("Using default prompt when using multiple dataset is incorrect.")
        
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            effort_history=self.effort_history,
            prompt_from_task=(self.default_prompt is None),
            max_episodes = self.max_episodes,
            rcs_sample_enable = self.rcs_sample_enable
        )

    
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        data_effort_history = getattr(self.data, "effort_history", None)
        if not data_effort_history and hasattr(self.data, "base_config") and self.data.base_config is not None:
            data_effort_history = getattr(self.data.base_config, "effort_history", None)

        if (
            data_effort_history
            and getattr(self.model, "effort_type", None)
            in (EffortType.LLM_HIS_C, EffortType.EXPERT_HIS_C, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT)
        ):
            object.__setattr__(
                self.model,
                "effort_dim_in",
                self.model.effort_dim * len(data_effort_history),
            ) 
        elif data_effort_history and getattr(self.model, "effort_dim", None) is not None:
            object.__setattr__(
                self.model,
                "effort_dim_in",
                self.model.effort_dim,
            )

# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
     TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(action_horizon=10),
        # Here you define the dataset you are training odddn. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=False, # The physical-intelligent/libero dataset already uses the delta action, so no additional transformation is needed.
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
        save_interval=10000,
        keep_period=10000,
        # ema_decay=None, # 开启以节约显存
    ),
    # TrainConfig(
    #     name="pi05_libero",
    #     model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    #     data=LeRobotLiberoDataConfig(
    #         repo_id="physical-intelligence/libero",
    #         base_config=DataConfig(prompt_from_task=True),
    #         extra_delta_transform=False,
    #     ),
    #     batch_size=32,
    #     lr_schedule=_optimizer.CosineDecaySchedule(
    #         warmup_steps=10_000,
    #         peak_lr=5e-5,
    #         decay_steps=1_000_000,
    #         decay_lr=5e-5,
    #     ),
    #     optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    #     ema_decay=0.999,
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    #     save_interval=10000,
    #     keep_period=10000,
    #     num_train_steps=30_000,
    # ),
      TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        save_interval=10000,
        keep_period=10000,
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_latent_flow_depth_teachers_libero",
        model=pi0_config.Pi0LatentFlowDepthTeachersConfig(
            action_horizon=10,
            effort_type=EffortType.NO,
            effort_dim=None,
            future_force_align_loss_weight=0.0,
            future_flow_align_loss_weight=1.0,
            future_depth_align_loss_weight=1.0,
            flow_token_count=16,
            depth_token_count=16,
            student_future_query_noise_scale_max=0.3,
            student_future_query_noise_start_ratio=0.3,
            student_future_query_noise_end_ratio=0.7,
            use_future_rgb_instead_of_flow=False,
        ),
        data=LeRobotLiberoFlowDepthDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=False,
        ),
        # weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_base/params"),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        save_interval=10000,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_latent_flow_depth_teachers_libero",
        model=pi0_config.Pi0LatentFlowDepthTeachersConfig(
            pi05=True,
            discrete_state_input=False,
            action_horizon=10,
            effort_type=EffortType.NO,
            effort_dim=None,
            future_force_align_loss_weight=0.0,
            future_flow_align_loss_weight=1.0,
            future_depth_align_loss_weight=1.0,
            flow_token_count=16,
            depth_token_count=16,
            student_future_query_noise_scale_max=0.3,
            student_future_query_noise_start_ratio=0.3,
            student_future_query_noise_end_ratio=0.7,
            use_future_rgb_instead_of_flow=False,
        ),
        data=LeRobotLiberoFlowDepthDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        num_train_steps=30_000,
        save_interval=10000,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi0_seer_0409",
        model=pi0_config.Pi0SeerConfig(
            action_horizon=32,
            effort_type=EffortType.EXPERT_HIS_C_FUT,
            effort_dim=6,
            foreseen_token_count_per_view=10,
            future_rgb_step=32,
            image_decoder_patch_size=16,
            image_decoder_input_size=224,
            future_image_loss_weight=0.1,
            use_future_rgb_instead_of_flow=True,
        ),
        data=LeRobotTaVLADataConfig(
            repo_id="llly/all_0409_stage_flow",
            effort_history=tuple((4 * i - 36 for i in range(10))),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        batch_size=16,
        save_interval=15000,
        keep_period=15000,
    ),
     TrainConfig(
        name="pi0_tavla_0409",
        model=pi0_config.Pi0TaVLAConfig(
            action_horizon=32,
            # effort_type=EffortType.EXPERT_HIS_C,
            effort_type=EffortType.EXPERT_HIS_C_FUT,
            effort_dim=6,  # 6-axis force sensor
        ),
        data=LeRobotTaVLADataConfig(
            # repo_id="llly/all_0409", # Placeholder, replace with your actual repo_id
            repo_id="llly/vga_0525", # Placeholder, replace with your actual repo_id
            effort_history=tuple((4*i-36 for i in range(10))), # sample 10 frames in 2s, assume fps =20
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=False, # Yuanluo actions are absolute
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_base/params"),
        num_train_steps=30_000, # Default to 30k steps, adjust as needed
        # num_workers=8,
        batch_size=16,
        save_interval=15000,
        keep_period=15000,
        # ema_decay = None # 节省显存
    ),
    TrainConfig(
        name="pi0_latent_flow",
        model=pi0_config.Pi0LatentFlowConfig(
            action_horizon=32,
            effort_type=EffortType.MOT,
            effort_dim=6,  # 6-axis force sensor
            # new parms
            force_input_frames=10,
            distill_layer_indices=(8, 12, 16),
            future_force_align_loss_weight=0.5,
            future_flow_align_loss_weight=0.5,
        ),
        data=LeRobotOptimalFlowDataConfig(
            repo_id="llly/all_0409_stage_flow", # Placeholder, replace with your actual repo_id
            effort_history=tuple(list((4 * i - 36 for i in range(10))) + list(range(1, 33))),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=False, # Yuanluo actions are absolute
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_base/params"),
        num_train_steps=30_000, # Default to 30k steps, adjust as needed
        # num_workers=8,
        batch_size=16,
        save_interval=15000,
        keep_period=15000,
        ema_decay = None # 节省显存
    ),
    TrainConfig(
        name="pi0_latent_flow_multiview",
        model=pi0_config.Pi0LatentFlowConfig(
            action_horizon=32,
            effort_type=EffortType.MOT,
            effort_dim=6,  # 6-axis force sensor
            force_input_frames=10,
            distill_layer_indices=(8, 12, 16),
            future_force_align_loss_weight=0.5,
            future_flow_align_loss_weight=0.5,
            use_future_rgb_instead_of_flow = False
        ),
        data=LeRobotOptimalFlowDataConfig(
            # This multiview config expects both
            # `observation.future_flow.base_0_rgb` and
            # `observation.future_flow.left_wrist_0_rgb` in the dataset.
            repo_id="llly/all_0409_stage_flow", # Placeholder, replace with your actual repo_id
            effort_history=tuple(list((4 * i - 36 for i in range(10))) + list(range(1, 33))),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=False, # Yuanluo actions are absolute
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_base/params"),
        # weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_latent_flow_multiview/pi0_latent_flow_multiview/29999/params"),
        num_train_steps=30_000, # Default to 30k steps, adjust as needed
        # num_workers=8,
        batch_size=16,
        save_interval=15000,
        keep_period=15000,
        ema_decay = None # 节省显存
    ),
    TrainConfig(
        name="pi0_latent_rgb_multiview",
        model=pi0_config.Pi0LatentFlowConfig(
            action_horizon=32,
            effort_type=EffortType.MOT,
            effort_dim=6,  # 6-axis force sensor
            force_input_frames=10,
            distill_layer_indices=(8, 12, 16),
            future_force_align_loss_weight=0.5,
            future_flow_align_loss_weight=0.5,
            use_future_rgb_instead_of_flow = True
        ),
        data=LeRobotOptimalFlowDataConfig(
            # This multiview config expects both
            # `observation.future_flow.base_0_rgb` and
            # `observation.future_flow.left_wrist_0_rgb` in the dataset.
            repo_id="llly/all_0409_stage_flow", # Placeholder, replace with your actual repo_id
            effort_history=tuple(list((4 * i - 36 for i in range(10))) + list(range(1, 33))),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=False, # Yuanluo actions are absolute
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_base/params"),
        # weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_latent_flow_multiview/pi0_latent_flow_multiview/29999/params"),
        num_train_steps=30_000, # Default to 30k steps, adjust as needed
        # num_workers=8,
        batch_size=16,
        save_interval=15000,
        keep_period=15000,
        ema_decay = None # 节省显存
    ),
    TrainConfig(
        name="pi0_latent_flow_noise",
        model=pi0_config.Pi0LatentFlowConfig(
            action_horizon=32,
            effort_type=EffortType.MOT,
            effort_dim=6,  # 6-axis force sensor
            # new parms
            force_input_frames=10,
            distill_layer_indices=(8, 12, 16),
            future_force_align_loss_weight=0.5,
            future_flow_align_loss_weight=0.5,
            student_future_query_noise_scale_max=0.3,
            student_future_query_noise_start_ratio=0.3,
            student_future_query_noise_end_ratio=0.7,
            use_future_rgb_instead_of_flow = False
        ),
        data=LeRobotOptimalFlowDataConfig(
            repo_id="llly/vga_0525_flow", # Placeholder, replace with your actual repo_id
            # repo_id="llly/all_0409_stage_flow", # Placeholder, replace with your actual repo_id
            effort_history=tuple(list((4 * i - 36 for i in range(10))) + list(range(1, 33))),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=False, # Yuanluo actions are absolute
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_base/params"),
        num_train_steps=30_000, # Default to 30k steps, adjust as needed
        # num_workers=8,
        batch_size=16,
        save_interval=15000,
        keep_period=15000,
        ema_decay = None # 节省显存
    ),
    TrainConfig(
        name="pi0_latent_future_rgb",
        model=pi0_config.Pi0LatentFlowConfig(
            action_horizon=32,
            effort_type=EffortType.MOT,
            effort_dim=6,  # 6-axis force sensor
            force_input_frames=10,
            distill_layer_indices=(8, 12, 16),
            future_force_align_loss_weight=0.5,
            future_flow_align_loss_weight=0.5,
            use_future_rgb_instead_of_flow=True,
            future_rgb_step=32,
        ),
        data=LeRobotOptimalFlowDataConfig(
            repo_id="llly/all_0409_stage_flow", # Placeholder, replace with your actual repo_id
            effort_history=tuple(list((4 * i - 36 for i in range(10))) + list(range(1, 33))),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=False, # Yuanluo actions are absolute
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_base/params"),
        num_train_steps=30_000, # Default to 30k steps, adjust as needed
        # num_workers=8,
        batch_size=16,
        save_interval=15000,
        keep_period=15000,
        ema_decay = None # 节省显存
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
