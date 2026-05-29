import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import model_tavla as _model_tavla
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

from openpi.shared.effort_type import EffortType

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    
    effort_type: str | None = None
    effort_dim: int | None = None
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_effort_input: bool = None  # type: ignore
    
    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class Pi0TaVLAConfig(Pi0Config):
    
    force_fast_tokenizer_path: str | None = None
    force_fast_vocab_size: int = 4096
    force_fast_action_dim: int = 8
    force_data_path: str | None = None # Path to force data (numpy array) for FAST tokenizer training
    force_data_repo_id: str | None = None # LeRobot repo id for force data for FAST tokenizer training

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0_tavla import Pi0TaVLA
        return Pi0TaVLA(self, rngs=nnx.Rngs(rng))

    @override
    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.effort_type in {EffortType.LLM_HIS_Lang, }:
            object.__setattr__(self, "max_token_len", 300 if self.pi05 else 48)
            object.__setattr__(self, "discrete_effort_input", True)


@dataclasses.dataclass(frozen=True)
class Pi0SeerConfig(Pi0TaVLAConfig):
    foreseen_token_count_per_view: int = 10
    future_image_views: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb")
    future_rgb_step: int = 0
    image_decoder_patch_size: int = 16
    image_decoder_input_size: int = 224
    future_image_loss_weight: float = 0.1
    predict_future_image_during_inference: bool = False
    use_future_rgb_instead_of_flow: bool = True
    normalize_future_patch_targets: bool = True

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0_seer import Pi0Seer

        return Pi0Seer(self, rngs=nnx.Rngs(rng))

    @override
    def __post_init__(self):
        super().__post_init__()
        if self.foreseen_token_count_per_view <= 0:
            raise ValueError("foreseen_token_count_per_view must be positive.")
        if self.image_decoder_patch_size <= 0:
            raise ValueError("image_decoder_patch_size must be positive.")
        if self.image_decoder_input_size % self.image_decoder_patch_size != 0:
            raise ValueError(
                "image_decoder_input_size must be divisible by image_decoder_patch_size, "
                f"got {self.image_decoder_input_size} and {self.image_decoder_patch_size}."
            )
        if not 0 <= self.future_rgb_step <= self.action_horizon:
            raise ValueError(
                f"future_rgb_step must satisfy 0 <= future_rgb_step <= action_horizon={self.action_horizon}, "
                f"got {self.future_rgb_step}."
            )

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model_tavla.Observation, _model_tavla.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        effort_dim = self.effort_dim if self.effort_dim is not None else self.action_dim
        effort_dim_in = getattr(self, "effort_dim_in", effort_dim)
        effort_steps = max(1, effort_dim_in // max(effort_dim, 1))

        with at.disable_typechecking():
            observation_spec = _model_tavla.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                effort=jax.ShapeDtypeStruct([batch_size, effort_steps, effort_dim], jnp.float32),
                future_rgb_img=image_spec,
                future_wrist_rgb_img=image_spec,
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec




@dataclasses.dataclass(frozen=True)
class Pi0LatentFlowConfig(Pi0Config):
    force_input_frames: int = 10
    teacher_action_loss_weight: float = 1.0
    student_action_loss_weight: float = 1.0
    distill_layer_indices: tuple = (8,12,16)
    future_force_align_loss_weight: float = 0.1
    future_flow_align_loss_weight: float = 0.1
    distill_projector_hidden_dim: int | None = None
    flow_token_count: int = 16
    flow_vae_name: str = "stabilityai/sdxl-vae"
    flow_vae_latent_channels: int = 4
    use_future_rgb_instead_of_flow: bool = False
    future_rgb_step: int = 0
    student_future_query_noise_scale_max: float = 0
    student_future_query_noise_start_ratio: float = 0
    student_future_query_noise_end_ratio: float = 0

    @override
    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.future_rgb_step <= self.action_horizon:
            raise ValueError(
                f"future_rgb_step must satisfy 0 <= future_rgb_step <= action_horizon={self.action_horizon}, "
                f"got {self.future_rgb_step}."
            )
        if self.student_future_query_noise_scale_max < 0.0:
            raise ValueError(
                "student_future_query_noise_scale_max must be non-negative, "
                f"got {self.student_future_query_noise_scale_max}."
            )
        if not 0.0 <= self.student_future_query_noise_start_ratio <= 1.0:
            raise ValueError(
                "student_future_query_noise_start_ratio must satisfy 0.0 <= p <= 1.0, "
                f"got {self.student_future_query_noise_start_ratio}."
            )
        if not 0.0 <= self.student_future_query_noise_end_ratio <= 1.0:
            raise ValueError(
                "student_future_query_noise_end_ratio must satisfy 0.0 <= p <= 1.0, "
                f"got {self.student_future_query_noise_end_ratio}."
            )
        if self.student_future_query_noise_start_ratio > self.student_future_query_noise_end_ratio:
            raise ValueError(
                "student_future_query_noise_start_ratio must be <= student_future_query_noise_end_ratio, "
                f"got start={self.student_future_query_noise_start_ratio}, "
                f"end={self.student_future_query_noise_end_ratio}."
            )

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0_latent_flow import Pi0LatentFlow
        return Pi0LatentFlow(self, rngs=nnx.Rngs(rng))


@dataclasses.dataclass(frozen=True)
class Pi0LatentFlowDepthTeachersConfig(Pi0LatentFlowConfig):
    flow_teacher_expert_variant: _gemma.Variant = "gemma_300m"
    depth_teacher_expert_variant: _gemma.Variant = "gemma_300m"
    flow_teacher_action_loss_weight: float = 1.0
    depth_teacher_action_loss_weight: float = 1.0
    future_depth_align_loss_weight: float = 0.1
    depth_token_count: int = 16
    depth_distill_projector_hidden_dim: int | None = None

    @override
    def __post_init__(self):
        super().__post_init__()
        if self.flow_token_count <= 0:
            raise ValueError(f"flow_token_count must be positive, got {self.flow_token_count}.")
        if self.depth_token_count <= 0:
            raise ValueError(f"depth_token_count must be positive, got {self.depth_token_count}.")
        if self.future_depth_align_loss_weight < 0.0:
            raise ValueError(
                "future_depth_align_loss_weight must be non-negative, "
                f"got {self.future_depth_align_loss_weight}."
            )

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0_latent_flow_depth_teachers import Pi0LatentFlowDepthTeachers

        return Pi0LatentFlowDepthTeachers(self, rngs=nnx.Rngs(rng))
