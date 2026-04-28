import einops
from diffusers import FlaxAutoencoderKL
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import gemma as _gemma
from openpi.models import model_tavla as _model
from openpi.models import pi0_config
from openpi.models import siglip as _siglip
from openpi.models.pi0_tavla import make_attn_mask, posemb_sincos
from openpi.shared import array_typing as at
from openpi.shared.effort_type import EffortType


_FLOW_VAE_REGISTRY: dict[str, tuple[FlaxAutoencoderKL, at.Params]] = {}


def register_flow_vae(model_name: str, flow_vae: FlaxAutoencoderKL, flow_vae_params: at.Params) -> None:
    _FLOW_VAE_REGISTRY[model_name] = (flow_vae, flow_vae_params)


def preload_flow_vae(model_name: str) -> None:
    if model_name in _FLOW_VAE_REGISTRY:
        return
    flow_vae, flow_vae_params = FlaxAutoencoderKL.from_pretrained(
        model_name,
        from_pt=True,
        dtype=jnp.float32,
    )
    register_flow_vae(model_name, flow_vae, flow_vae_params)


def _get_flow_vae(model_name: str) -> tuple[FlaxAutoencoderKL, at.Params]:
    if model_name not in _FLOW_VAE_REGISTRY:
        raise ValueError(
            f"Flow VAE '{model_name}' has not been preloaded. "
            "Call preload_flow_vae(...) in Python before initializing training."
        )
    return _FLOW_VAE_REGISTRY[model_name]


class Pi0LatentFlow(_model.BaseModel):
    """Standalone dual-expert model with future-force and future-flow alignment."""

    def __init__(self, config: pi0_config.Pi0LatentFlowConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        effort_dim_in = getattr(config, "effort_dim_in", None)
        if effort_dim_in is None:
            effort_dim_in = config.effort_dim
        if effort_dim_in is None or int(effort_dim_in) <= 0:
            raise ValueError("Pi0MORDualAlignForceFlow requires a positive `effort_dim_in`.")

        self.effort_dim_in = int(effort_dim_in)
        self.effort_dim = int(config.effort_dim if config.effort_dim is not None else self.effort_dim_in)
        self.pi05 = config.pi05
        self.effort_type = config.effort_type

        self.force_input_frames = int(config.force_input_frames)
        self.distill_layer_indices = tuple(int(i) for i in config.distill_layer_indices)
        self.student_action_loss_weight = float(config.student_action_loss_weight)
        self.teacher_action_loss_weight = float(config.teacher_action_loss_weight)
        self.future_force_align_loss_weight = float(config.future_force_align_loss_weight)
        self.future_flow_align_loss_weight = float(config.future_flow_align_loss_weight)
        self.flow_token_count = int(config.flow_token_count)
        self.future_flow_channels = 3
        self.flow_vae_name = getattr(config, "flow_vae_name", "stabilityai/sdxl-vae")
        self.use_future_rgb_instead_of_flow = bool(config.use_future_rgb_instead_of_flow)
        self.future_rgb_step = int(config.future_rgb_step)
        self.student_future_query_noise_prob_max = float(config.student_future_query_noise_prob_max)
        self.student_future_query_noise_start_ratio = float(config.student_future_query_noise_start_ratio)
        self.student_future_query_noise_end_ratio = float(config.student_future_query_noise_end_ratio)
        self.student_future_query_noise_scale = float(config.student_future_query_noise_scale)
        self.uses_train_progress = True
        self._debug_lengths_logged = False


        paligemma_config = _gemma.get_config(config.paligemma_variant)
        student_config = _gemma.get_config(config.action_expert_variant)
        teacher_variant = getattr(config, "force_expert_variant", config.action_expert_variant)
        teacher_config = _gemma.get_config(teacher_variant)
        self.student_width = int(student_config.width)
        self.teacher_width = int(teacher_config.width)
        self.distill_projector_hidden_dim = int(
            config.distill_projector_hidden_dim
            if config.distill_projector_hidden_dim is not None
            else teacher_config.width
        )

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, student_config, teacher_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True, True] if config.pi05 else [False, False, False],
        )

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        if config.pi05:
            self.student_time_mlp_in = nnx.Linear(student_config.width, student_config.width, rngs=rngs)
            self.student_time_mlp_out = nnx.Linear(student_config.width, student_config.width, rngs=rngs)
            self.teacher_time_mlp_in = nnx.Linear(teacher_config.width, teacher_config.width, rngs=rngs)
            self.teacher_time_mlp_out = nnx.Linear(teacher_config.width, teacher_config.width, rngs=rngs)
        else:
            self.student_time_mlp_in = nnx.Linear(2 * student_config.width, student_config.width, rngs=rngs)
            self.student_time_mlp_out = nnx.Linear(student_config.width, student_config.width, rngs=rngs)
            self.teacher_time_mlp_in = nnx.Linear(2 * teacher_config.width, teacher_config.width, rngs=rngs)
            self.teacher_time_mlp_out = nnx.Linear(teacher_config.width, teacher_config.width, rngs=rngs)

        self.state_proj_student = nnx.Linear(config.action_dim, student_config.width, rngs=rngs)
        self.state_proj_teacher = nnx.Linear(config.action_dim, teacher_config.width, rngs=rngs)
        self.action_in_proj_student = nnx.Linear(config.action_dim, student_config.width, rngs=rngs)
        self.action_in_proj_teacher = nnx.Linear(config.action_dim, teacher_config.width, rngs=rngs)
        self.action_out_proj_student = nnx.Linear(student_config.width, config.action_dim, rngs=rngs)
        self.action_out_proj_teacher = nnx.Linear(teacher_config.width, config.action_dim, rngs=rngs)

        history_dim = self.force_input_frames * self.effort_dim_in
        future_dim = config.action_horizon * self.effort_dim_in
        self.history_force_proj_student = nnx.Linear(history_dim, student_config.width, rngs=rngs)
        self.history_force_proj_teacher = nnx.Linear(history_dim, teacher_config.width, rngs=rngs)
        self.future_force_proj_teacher = nnx.Linear(future_dim, teacher_config.width, rngs=rngs)
        self.student_query = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (student_config.width,), dtype=jnp.float32)
        )
        self.student_future_mask_token = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (student_config.width,), dtype=jnp.float32)
        )
        self.prompt_distill_proj_in = nnx.Linear(
            student_config.width, self.distill_projector_hidden_dim, rngs=rngs
        )
        self.prompt_distill_proj_out = nnx.Linear(
            self.distill_projector_hidden_dim, teacher_config.width, rngs=rngs
        )

        self.student_future_flow_query = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (self.flow_token_count, self.student_width),
                dtype=jnp.float32,
            )
        )
        self.teacher_future_flow_query = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (self.flow_token_count, self.teacher_width),
                dtype=jnp.float32,
            )
        )
        self.flow_distill_proj_in = nnx.Linear(
            self.student_width, self.distill_projector_hidden_dim, rngs=rngs
        )
        self.flow_distill_proj_out = nnx.Linear(
            self.distill_projector_hidden_dim, self.teacher_width, rngs=rngs
        )

        self.flow_vae_latent_channels = int(config.flow_vae_latent_channels)
        self.flow_vae_patch_merge_factor = 2
        self.flow_vae_proj_in = nnx.Linear(
            self.flow_vae_latent_channels * (self.flow_vae_patch_merge_factor ** 2),
            self.teacher_width,
            rngs=rngs,
        )
        self.flow_vae_proj_out = nnx.Linear(self.teacher_width, self.teacher_width, rngs=rngs)
        self.flow_vae_norm = nnx.LayerNorm(num_features=self.teacher_width, rngs=rngs)
        self.flow_vae_query_proj = nnx.Linear(self.teacher_width, self.teacher_width, rngs=rngs)
        self.flow_vae_key_proj = nnx.Linear(self.teacher_width, self.teacher_width, rngs=rngs)
        self.flow_vae_value_proj = nnx.Linear(self.teacher_width, self.teacher_width, rngs=rngs)
        self.flow_token_embedding = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (self.flow_token_count, self.teacher_width),
                dtype=jnp.float32,
            )
        )

        self.deterministic = True

    def _pad_or_crop_effort(
        self,
        effort: at.Float[at.Array, "b t d"],
        steps: int,
        *,
        from_end: bool,
    ) -> at.Float[at.Array, "b t d"]:
        if effort.shape[1] >= steps:
            return effort[:, -steps:, :] if from_end else effort[:, :steps, :]
        if effort.shape[1] == 0:
            raise ValueError("Pi0MORDualAlignForceFlow received an empty effort sequence.")
        pad_source = effort[:, :1, :] if from_end else effort[:, -1:, :]
        pad = jnp.repeat(pad_source, steps - effort.shape[1], axis=1)
        return jnp.concatenate([pad, effort], axis=1) if from_end else jnp.concatenate([effort, pad], axis=1)

    def _split_effort(
        self,
        observation: _model.Observation,
        *,
        require_future: bool,
        dtype: at.DTypeLike,
    ) -> tuple[at.Float[at.Array, "b h e"], at.Float[at.Array, "b f e"] | None]:
        if observation.effort is None:
            raise ValueError("Pi0MORDualAlignForceFlow requires `observation.effort`.")
        effort = jnp.asarray(observation.effort, dtype=dtype)
        if effort.ndim != 3:
            raise ValueError(f"Expected effort with shape [B, T, D], got {effort.shape}.")

        future_effort = None
        if effort.shape[1] > self.action_horizon:
            history_effort = effort[:, : effort.shape[1] - self.action_horizon, :]
            future_effort = effort[:, -self.action_horizon :, :]
        else:
            history_effort = effort

        if require_future and future_effort is None:
            raise ValueError(
                "Teacher training requires merged history+future effort. "
                f"Expected more than action_horizon={self.action_horizon} effort steps, got {effort.shape[1]}."
            )

        history_effort = self._pad_or_crop_effort(history_effort, self.force_input_frames, from_end=True)
        if future_effort is not None:
            future_effort = self._pad_or_crop_effort(future_effort, self.action_horizon, from_end=False)
        return history_effort, future_effort

    def _project_history_force_student(
        self, history_effort: at.Float[at.Array, "b h e"]
    ) -> at.Float[at.Array, "b 1 d"]:
        hidden = self.history_force_proj_student(einops.rearrange(history_effort, "b h e -> b (h e)"))
        return hidden[:, None, :]

    def _project_history_force_teacher(
        self, history_effort: at.Float[at.Array, "b h e"]
    ) -> at.Float[at.Array, "b 1 d"]:
        hidden = self.history_force_proj_teacher(einops.rearrange(history_effort, "b h e -> b (h e)"))
        return hidden[:, None, :]

    def _project_future_force_teacher(
        self, future_effort: at.Float[at.Array, "b h e"]
    ) -> at.Float[at.Array, "b 1 d"]:
        hidden = self.future_force_proj_teacher(einops.rearrange(future_effort, "b h e -> b (h e)"))
        return hidden[:, None, :]

    def _student_query_token(self, batch_size: int, dtype: jnp.dtype) -> at.Float[at.Array, "b 1 d"]:
        query = jnp.asarray(self.student_query.value, dtype=dtype)
        return jnp.broadcast_to(query[None, None, :], (batch_size, 1, query.shape[0]))

    def _student_query_noise_prob(self, train_progress: at.Float[at.Array, ""] | float | None) -> at.Float[at.Array, ""]:
        if train_progress is None:
            return jnp.asarray(0.0, dtype=jnp.float32)
        progress = jnp.clip(jnp.asarray(train_progress, dtype=jnp.float32), 0.0, 1.0)
        start = jnp.asarray(self.student_future_query_noise_start_ratio, dtype=jnp.float32)
        end = jnp.asarray(self.student_future_query_noise_end_ratio, dtype=jnp.float32)
        max_prob = jnp.asarray(self.student_future_query_noise_prob_max, dtype=jnp.float32)
        ramp = (progress - start) / jnp.maximum(end - start, 1e-6)
        return max_prob * jnp.clip(ramp, 0.0, 1.0)

    def _student_future_query_tokens(
        self,
        batch_size: int,
        dtype: jnp.dtype,
        *,
        train: bool,
        noise_rng: at.KeyArrayLike | None,
        train_progress: at.Float[at.Array, ""] | float | None = None,
        query_noise_prob: at.Float[at.Array, ""] | float | None = None,
        query_noise_scale: at.Float[at.Array, ""] | float | None = None,
    ) -> tuple[
        at.Float[at.Array, "b 1 d"],
        at.Float[at.Array, "b n d"],
        at.Bool[at.Array, "b"],
        at.Bool[at.Array, "b n"],
        at.Float[at.Array, "b"],
    ]:
        future_force_query = self._student_query_token(batch_size, dtype)
        future_flow_queries = self._student_future_flow_tokens(batch_size, dtype)
        noise_prob = (
            self._student_query_noise_prob(train_progress)
            if query_noise_prob is None
            else jnp.asarray(query_noise_prob, dtype=jnp.float32)
        )
        noise_prob = jnp.clip(noise_prob, 0.0, 1.0)
        noise_scale = (
            jnp.asarray(self.student_future_query_noise_scale, dtype=dtype)
            if query_noise_scale is None
            else jnp.asarray(query_noise_scale, dtype=dtype)
        )
        if not train and query_noise_prob is None:
            return (
                future_force_query,
                future_flow_queries,
                jnp.ones((batch_size,), dtype=jnp.bool_),
                jnp.ones((batch_size, self.flow_token_count), dtype=jnp.bool_),
                jnp.zeros((batch_size,), dtype=jnp.float32),
            )
        if query_noise_prob is None and float(self.student_future_query_noise_prob_max) <= 0.0:
            return (
                future_force_query,
                future_flow_queries,
                jnp.ones((batch_size,), dtype=jnp.bool_),
                jnp.ones((batch_size, self.flow_token_count), dtype=jnp.bool_),
                jnp.zeros((batch_size,), dtype=jnp.float32),
            )
        if noise_rng is None:
            raise ValueError("noise_rng is required when training with student future query noise enabled.")

        force_rng, flow_rng, force_noise_rng, flow_noise_rng = jax.random.split(noise_rng, 4)
        force_noise_mask = jax.random.bernoulli(force_rng, p=noise_prob, shape=(batch_size, 1))
        flow_noise_mask = jax.random.bernoulli(flow_rng, p=noise_prob, shape=(batch_size, self.flow_token_count))

        force_noise = noise_scale * jax.random.normal(force_noise_rng, future_force_query.shape, dtype=dtype)
        flow_noise = noise_scale * jax.random.normal(flow_noise_rng, future_flow_queries.shape, dtype=dtype)

        future_force_query = jnp.where(force_noise_mask[:, :, None], force_noise, future_force_query)
        future_flow_queries = jnp.where(flow_noise_mask[:, :, None], flow_noise, future_flow_queries)
        force_clean_mask = jnp.logical_not(jnp.squeeze(force_noise_mask, axis=1))
        flow_clean_mask = jnp.logical_not(flow_noise_mask)
        noised_token_count = jnp.squeeze(force_noise_mask.astype(jnp.float32), axis=1) + jnp.sum(
            flow_noise_mask.astype(jnp.float32), axis=1
        )
        total_token_count = jnp.asarray(1 + self.flow_token_count, dtype=jnp.float32)
        return future_force_query, future_flow_queries, force_clean_mask, flow_clean_mask, noised_token_count / total_token_count

    def _project_prompt_distill(self, hidden: at.Float[at.Array, "b t d"]) -> at.Float[at.Array, "b t d"]:
        hidden = self.prompt_distill_proj_in(hidden)
        hidden = nnx.swish(hidden)
        return self.prompt_distill_proj_out(hidden)

    def _project_flow_distill(self, hidden: at.Float[at.Array, "b t d"]) -> at.Float[at.Array, "b t d"]:
        hidden = self.flow_distill_proj_in(hidden)
        hidden = nnx.swish(hidden)
        return self.flow_distill_proj_out(hidden)

    @staticmethod
    def _apply_loss_mask(
        losses: at.Float[at.Array, "b"],
        keep_mask: at.Bool[at.Array, "b"],
    ) -> at.Float[at.Array, "b"]:
        return jnp.where(keep_mask, losses, jnp.zeros_like(losses))

    @staticmethod
    def _masked_mean(
        losses: at.Float[at.Array, "b"],
        keep_mask: at.Bool[at.Array, "b"],
    ) -> at.Float[at.Array, ""]:
        weights = keep_mask.astype(losses.dtype)
        denom = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=losses.dtype))
        return jnp.sum(losses * weights) / denom

    @staticmethod
    def _cosine_distance(
        lhs: at.Float[at.Array, "b t d"],
        rhs: at.Float[at.Array, "b t d"],
    ) -> at.Float[at.Array, " b"]:
        lhs = lhs.astype(jnp.float32)
        rhs = rhs.astype(jnp.float32)
        lhs_norm = lhs / jnp.sqrt(jnp.sum(jnp.square(lhs), axis=-1, keepdims=True) + 1e-6)
        rhs_norm = rhs / jnp.sqrt(jnp.sum(jnp.square(rhs), axis=-1, keepdims=True) + 1e-6)
        cosine = jnp.sum(lhs_norm * rhs_norm, axis=-1)
        return jnp.mean(1.0 - cosine, axis=-1)

    @staticmethod
    def _cosine_distance_masked(
        lhs: at.Float[at.Array, "b t d"],
        rhs: at.Float[at.Array, "b t d"],
        token_mask: at.Bool[at.Array, "b t"],
    ) -> at.Float[at.Array, " b"]:
        lhs = lhs.astype(jnp.float32)
        rhs = rhs.astype(jnp.float32)
        lhs_norm = lhs / jnp.sqrt(jnp.sum(jnp.square(lhs), axis=-1, keepdims=True) + 1e-6)
        rhs_norm = rhs / jnp.sqrt(jnp.sum(jnp.square(rhs), axis=-1, keepdims=True) + 1e-6)
        cosine = jnp.sum(lhs_norm * rhs_norm, axis=-1)
        losses = 1.0 - cosine
        weights = token_mask.astype(losses.dtype)
        denom = jnp.maximum(jnp.sum(weights, axis=-1), jnp.asarray(1.0, dtype=losses.dtype))
        return jnp.sum(losses * weights, axis=-1) / denom

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _embed_action_tokens(
        self,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        *,
        expert: str,
    ) -> tuple[at.Float[at.Array, "b ah d"], at.Float[at.Array, "b d"] | None]:
        if expert == "student":
            action_tokens = self.action_in_proj_student(noisy_actions)
            width = self.action_in_proj_student.out_features
            time_mlp_in = self.student_time_mlp_in
            time_mlp_out = self.student_time_mlp_out
        elif expert == "teacher":
            action_tokens = self.action_in_proj_teacher(noisy_actions)
            width = self.action_in_proj_teacher.out_features
            time_mlp_in = self.teacher_time_mlp_in
            time_mlp_out = self.teacher_time_mlp_out
        else:
            raise ValueError(f"Unknown expert: {expert}")

        time_emb = posemb_sincos(timestep, width, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            return action_tokens, time_emb

        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=noisy_actions.shape[1])
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = time_mlp_out(action_time_tokens)
        return action_time_tokens, None

    def _student_future_flow_tokens(self, batch_size: int, dtype: jnp.dtype) -> at.Float[at.Array, "b n d"]:
        query = jnp.asarray(self.student_future_flow_query.value, dtype=dtype)
        return jnp.broadcast_to(query[None, :, :], (batch_size, query.shape[0], query.shape[1]))

    def _build_suffix_ar_mask(self, action_len: int) -> at.Bool[at.Array, " s"]:
        return jnp.array(
            [False, False] + [True] + ([False] * self.flow_token_count) + [True] + ([False] * (action_len - 1))
        )

    @staticmethod
    def _restore_aux_images(
        processed: _model.Observation,
        original_flow_img: at.Array | None,
        original_wrist_flow_img: at.Array | None,
        original_future_rgb_img: at.Array | None,
        original_future_wrist_rgb_img: at.Array | None,
    ) -> _model.Observation:
        updates = {}
        if original_flow_img is not None:
            updates["flow_img"] = original_flow_img
        if original_wrist_flow_img is not None:
            updates["wrist_flow_img"] = original_wrist_flow_img
        if original_future_rgb_img is not None:
            updates["future_rgb_img"] = original_future_rgb_img
        if original_future_wrist_rgb_img is not None:
            updates["future_wrist_rgb_img"] = original_future_wrist_rgb_img
        if not updates:
            return processed
        return processed.replace(**updates)

    @staticmethod
    def _require_aux_image(
        image: at.Array | None,
        *,
        field_name: str,
        mode_name: str,
    ) -> at.Array:
        if image is None:
            raise ValueError(f"Pi0LatentFlow requires `{field_name}` for {mode_name}.")
        return image

    def _get_future_visual_images(self, obs: _model.Observation) -> tuple[at.Array, at.Array]:
        if self.use_future_rgb_instead_of_flow:
            return (
                self._require_aux_image(
                    obs.future_rgb_img,
                    field_name="observation.future_rgb_img",
                    mode_name="RGB ablation when `use_future_rgb_instead_of_flow=True`",
                ),
                self._require_aux_image(
                    obs.future_wrist_rgb_img,
                    field_name="observation.future_wrist_rgb_img",
                    mode_name="RGB ablation when `use_future_rgb_instead_of_flow=True`",
                ),
            )
        return (
            self._require_aux_image(
                obs.flow_img,
                field_name="observation.flow_img",
                mode_name="flow distillation",
            ),
            self._require_aux_image(
                obs.wrist_flow_img,
                field_name="observation.wrist_flow_img",
                mode_name="flow distillation",
            ),
        )

    def _encode_future_visual_image(self, image: at.Float[at.Array, "b h w c"]) -> at.Float[at.Array, "b s d"]:
        x = jnp.asarray(image, dtype=jnp.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected future visual image with shape [B, H, W, C], got {x.shape}.")
        if x.shape[-1] != self.future_flow_channels:
            raise ValueError(
                f"Expected future visual image with {self.future_flow_channels} channels, got shape={x.shape}."
            )
        # FlaxAutoencoderKL.encode expects BCHW input and internally converts to NHWC.
        x = jnp.transpose(x, (0, 3, 1, 2))

        flow_vae, flow_vae_params = _get_flow_vae(self.flow_vae_name)
        posterior = flow_vae.apply(
            {"params": flow_vae_params},
            x,
            deterministic=True,
            method=flow_vae.encode,
        ).latent_dist
        x = posterior.mode() * flow_vae.config.scaling_factor
        x = jax.lax.stop_gradient(x)
        patch = self.flow_vae_patch_merge_factor
        if x.shape[1] % patch != 0 or x.shape[2] % patch != 0:
            raise ValueError(
                "Future visual latent spatial size must be divisible by "
                f"{patch}, got shape={x.shape}."
            )
        x = einops.rearrange(
            x,
            "b (h ph) (w pw) c -> b h w (ph pw c)",
            ph=patch,
            pw=patch,
        )
        latent_tokens = einops.rearrange(x, "b h w c -> b (h w) c")
        latent_tokens = self.flow_vae_proj_in(latent_tokens)
        latent_tokens = nnx.swish(latent_tokens)
        latent_tokens = self.flow_vae_proj_out(latent_tokens)
        return self.flow_vae_norm(latent_tokens)

    def _compress_future_flows(self, obs: _model.Observation) -> at.Float[at.Array, "b n d"]:
        future_images = self._get_future_visual_images(obs)
        latent_tokens = jnp.concatenate(
            [self._encode_future_visual_image(image) for image in future_images],
            axis=1,
        )
        query = jnp.asarray(self.teacher_future_flow_query.value, dtype=latent_tokens.dtype)
        query = query + jnp.asarray(self.flow_token_embedding.value, dtype=latent_tokens.dtype)
        query = jnp.broadcast_to(query[None, :, :], (latent_tokens.shape[0], query.shape[0], query.shape[1]))
        query = self.flow_vae_query_proj(query)
        keys = self.flow_vae_key_proj(latent_tokens)
        values = self.flow_vae_value_proj(latent_tokens)

        logits = jnp.einsum("bqd,bkd->bqk", query, keys)
        logits = logits / jnp.sqrt(jnp.asarray(self.teacher_width, dtype=logits.dtype))
        attn = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(values.dtype)
        return jnp.einsum("bqk,bkd->bqd", attn, values)

    @at.typecheck
    def embed_student_suffix(
        self,
        obs: _model.Observation,
        history_effort: at.Float[at.Array, "b h e"],
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        *,
        train: bool = False,
        noise_rng: at.KeyArrayLike | None = None,
        train_progress: at.Float[at.Array, ""] | float | None = None,
        query_noise_prob: at.Float[at.Array, ""] | float | None = None,
        query_noise_scale: at.Float[at.Array, ""] | float | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
        at.Bool[at.Array, "b"],
        at.Bool[at.Array, "b n"],
        at.Float[at.Array, "b"],
    ]:
        history_token = self._project_history_force_student(history_effort)
        future_force_query, future_flow_queries, force_clean_mask, flow_clean_mask, noised_token_rate = self._student_future_query_tokens(
            obs.state.shape[0],
            history_token.dtype,
            train=train,
            noise_rng=noise_rng,
            train_progress=train_progress,
            query_noise_prob=query_noise_prob,
            query_noise_scale=query_noise_scale,
        )
        state_token = self.state_proj_student(obs.state)[:, None, :]
        action_tokens, adarms_cond = self._embed_action_tokens(noisy_actions, timestep, expert="student")

        tokens = jnp.concatenate(
            [history_token, state_token, future_force_query, future_flow_queries, action_tokens], axis=1
        )
        input_mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = self._build_suffix_ar_mask(action_tokens.shape[1])
        return tokens, input_mask, ar_mask, adarms_cond, force_clean_mask, flow_clean_mask, noised_token_rate

    @at.typecheck
    def embed_teacher_suffix(
        self,
        obs: _model.Observation,
        history_effort: at.Float[at.Array, "b h e"],
        future_effort: at.Float[at.Array, "b f e"],
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        history_token = self._project_history_force_teacher(history_effort)
        future_force_token = self._project_future_force_teacher(future_effort)
        future_flow_tokens = self._compress_future_flows(obs)
        state_token = self.state_proj_teacher(obs.state)[:, None, :]
        action_tokens, adarms_cond = self._embed_action_tokens(noisy_actions, timestep, expert="teacher")

        tokens = jnp.concatenate(
            [history_token, state_token, future_force_token, future_flow_tokens, action_tokens], axis=1
        )
        input_mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = self._build_suffix_ar_mask(action_tokens.shape[1])
        return tokens, input_mask, ar_mask, adarms_cond

    def _forward_train_joint_multilayer(
        self,
        *,
        prefix_tokens: at.Float[at.Array, "b p d"],
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_ar_mask: at.Bool[at.Array, " p"],
        student_suffix_tokens: at.Float[at.Array, "b a d"],
        student_suffix_mask: at.Bool[at.Array, "b a"],
        student_suffix_ar_mask: at.Bool[at.Array, " a"],
        student_adarms: at.Float[at.Array, "b d"] | None,
        teacher_suffix_tokens: at.Float[at.Array, "b f d"],
        teacher_suffix_mask: at.Bool[at.Array, "b f"],
        teacher_suffix_ar_mask: at.Bool[at.Array, " f"],
        teacher_adarms: at.Float[at.Array, "b d"] | None,
    ) -> tuple[
        at.Float[at.Array, "b a d"],
        at.Float[at.Array, "b f d"],
        tuple[at.Float[at.Array, "b a d"], ...],
        tuple[at.Float[at.Array, "b f d"], ...],
    ]:
        bsz = prefix_mask.shape[0]
        p_len = prefix_mask.shape[1]
        s_len = student_suffix_mask.shape[1]
        t_len = teacher_suffix_mask.shape[1]

        prefix_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
        student_attn = make_attn_mask(student_suffix_mask, student_suffix_ar_mask)
        teacher_attn = make_attn_mask(teacher_suffix_mask, teacher_suffix_ar_mask)

        student_to_prefix = einops.repeat(prefix_mask, "b p -> b s p", s=s_len)
        student_to_prefix = jnp.logical_and(student_to_prefix, student_suffix_mask[:, :, None])
        teacher_to_prefix = einops.repeat(prefix_mask, "b p -> b t p", t=t_len)
        teacher_to_prefix = jnp.logical_and(teacher_to_prefix, teacher_suffix_mask[:, :, None])

        prefix_row = jnp.concatenate(
            [
                prefix_attn,
                jnp.zeros((bsz, p_len, s_len), dtype=jnp.bool_),
                jnp.zeros((bsz, p_len, t_len), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        student_row = jnp.concatenate(
            [
                student_to_prefix,
                student_attn,
                jnp.zeros((bsz, s_len, t_len), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        teacher_row = jnp.concatenate(
            [
                teacher_to_prefix,
                jnp.zeros((bsz, t_len, s_len), dtype=jnp.bool_),
                teacher_attn,
            ],
            axis=-1,
        )
        full_attn = jnp.concatenate([prefix_row, student_row, teacher_row], axis=1)

        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_len = jnp.sum(prefix_mask, axis=-1)[:, None]
        student_positions = prefix_len + jnp.cumsum(student_suffix_mask, axis=-1) - 1
        teacher_positions = prefix_len + jnp.cumsum(teacher_suffix_mask, axis=-1) - 1
        positions = jnp.concatenate([prefix_positions, student_positions, teacher_positions], axis=1)

        (outputs, selected_layers), _ = self.PaliGemma.llm(
            [prefix_tokens, student_suffix_tokens, teacher_suffix_tokens],
            mask=full_attn,
            positions=positions,
            adarms_cond=[None, student_adarms, teacher_adarms],
            return_layer_indices=self.distill_layer_indices,
        )
        _, student_out, teacher_out = outputs
        student_layer_hiddens = tuple(layer[1] for layer in selected_layers)
        teacher_layer_hiddens = tuple(layer[2] for layer in selected_layers)
        return student_out, teacher_out, student_layer_hiddens, teacher_layer_hiddens

    def compute_loss_with_stats(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        train_progress: at.Float[at.Array, ""] | float | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng, query_noise_rng = jax.random.split(rng, 4)
        original_flow_img = observation.flow_img
        original_wrist_flow_img = observation.wrist_flow_img
        original_future_rgb_img = observation.future_rgb_img
        original_future_wrist_rgb_img = observation.future_wrist_rgb_img
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, effort_type=self.effort_type
        )
        observation = self._restore_aux_images(
            observation,
            original_flow_img,
            original_wrist_flow_img,
            original_future_rgb_img,
            original_future_wrist_rgb_img,
        )
        history_effort, future_effort = self._split_effort(observation, require_future=True, dtype=actions.dtype)
        if future_effort is None:
            raise ValueError("Pi0MORDualAlignForceFlow teacher training requires future effort.")

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t_action = time_expanded * noise + (1 - time_expanded) * actions
        u_t_action = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        (
            student_tokens,
            student_mask,
            student_ar_mask,
            student_adarms,
            force_clean_mask,
            flow_clean_mask,
            noised_token_rate,
        ) = self.embed_student_suffix(
            observation,
            history_effort,
            x_t_action,
            time,
            train=train,
            noise_rng=query_noise_rng,
            train_progress=train_progress,
        )
        teacher_tokens, teacher_mask, teacher_ar_mask, teacher_adarms = self.embed_teacher_suffix(
            observation, history_effort, future_effort, x_t_action, time
        )
        student_out, teacher_out, student_layer_hiddens, teacher_layer_hiddens = self._forward_train_joint_multilayer(
            prefix_tokens=prefix_tokens,
            prefix_mask=prefix_mask,
            prefix_ar_mask=prefix_ar_mask,
            student_suffix_tokens=student_tokens,
            student_suffix_mask=student_mask,
            student_suffix_ar_mask=student_ar_mask,
            student_adarms=student_adarms,
            teacher_suffix_tokens=teacher_tokens,
            teacher_suffix_mask=teacher_mask,
            teacher_suffix_ar_mask=teacher_ar_mask,
            teacher_adarms=teacher_adarms,
        )

        student_v = self.action_out_proj_student(student_out[:, -self.action_horizon :])
        teacher_v = self.action_out_proj_teacher(teacher_out[:, -self.action_horizon :])
        student_action_loss = jnp.mean(jnp.square(student_v - u_t_action), axis=(-2, -1))
        teacher_action_loss = jnp.mean(jnp.square(teacher_v - u_t_action), axis=(-2, -1))

        force_losses = []
        flow_losses = []
        future_force_slice = slice(2, 3)
        flow_slice = slice(3, 3 + self.flow_token_count)
        for student_hidden, teacher_hidden in zip(student_layer_hiddens, teacher_layer_hiddens, strict=True):
            student_force_hidden = self._project_prompt_distill(student_hidden[:, future_force_slice, :])
            teacher_force_hidden = jax.lax.stop_gradient(teacher_hidden[:, future_force_slice, :])
            force_losses.append(
                self._apply_loss_mask(
                    self._cosine_distance(student_force_hidden, teacher_force_hidden),
                    force_clean_mask,
                )
            )

            student_flow_hidden = self._project_flow_distill(student_hidden[:, flow_slice, :])
            teacher_flow_hidden = jax.lax.stop_gradient(teacher_hidden[:, flow_slice, :])
            flow_losses.append(
                self._cosine_distance_masked(
                    student_flow_hidden,
                    teacher_flow_hidden,
                    flow_clean_mask,
                )
            )

        raw_future_force_align_loss = jnp.mean(jnp.stack(force_losses, axis=0), axis=0)
        raw_future_flow_align_loss = jnp.mean(jnp.stack(flow_losses, axis=0), axis=0)
        future_force_align_loss = raw_future_force_align_loss
        future_flow_align_loss = raw_future_flow_align_loss

        total_loss = (
            self.student_action_loss_weight * student_action_loss
            + self.teacher_action_loss_weight * teacher_action_loss
            + self.future_force_align_loss_weight * future_force_align_loss
            + self.future_flow_align_loss_weight * future_flow_align_loss
        )
        stats = {
            "loss/student_action": student_action_loss,
            "loss/teacher_action": teacher_action_loss,
            "loss/distill_future_force": future_force_align_loss,
            "loss/distill_future_flow": future_flow_align_loss,
            "loss/distill_future_force_mean": jnp.mean(raw_future_force_align_loss),
            "loss/distill_future_flow_mean": jnp.mean(raw_future_flow_align_loss),
            "noise/student_future_query_token_rate": jnp.mean(noised_token_rate),
            "noise/student_future_query_prob": self._student_query_noise_prob(train_progress),
            "loss/total": total_loss,
        }
        return total_loss, stats

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        train_progress: at.Float[at.Array, ""] | float | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self.compute_loss_with_stats(rng, observation, actions, train=train, train_progress=train_progress)
        return loss

    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        debug_query_noise_prob: float | None = None,
        debug_query_noise_scale: float | None = None,
    ) -> _model.Actions:
        original_flow_img = observation.flow_img
        original_wrist_flow_img = observation.wrist_flow_img
        original_future_rgb_img = observation.future_rgb_img
        original_future_wrist_rgb_img = observation.future_wrist_rgb_img
        observation = _model.preprocess_observation(None, observation, train=False, effort_type=self.effort_type)
        observation = self._restore_aux_images(
            observation,
            original_flow_img,
            original_wrist_flow_img,
            original_future_rgb_img,
            original_future_wrist_rgb_img,
        )
        history_effort, _ = self._split_effort(observation, require_future=False, dtype=jnp.float32)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        action_noise_rng, query_noise_rng = jax.random.split(rng)
        if noise is None:
            noise = jax.random.normal(action_noise_rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            adarms_cond=[None, None, None],
        )

        def step(carry):
            x_t, time, step_rng = carry
            step_rng, iter_query_noise_rng = jax.random.split(step_rng)
            student_tokens, student_mask, student_ar_mask, student_adarms, *_ = self.embed_student_suffix(
                observation,
                history_effort,
                x_t,
                jnp.broadcast_to(time, batch_size),
                train=False,
                noise_rng=iter_query_noise_rng,
                query_noise_prob=debug_query_noise_prob,
                query_noise_scale=debug_query_noise_scale,
            )
            student_attn_mask = make_attn_mask(student_mask, student_ar_mask)
            prefix_to_student = einops.repeat(prefix_mask, "b p -> b s p", s=student_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_to_student, student_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(student_mask, axis=-1) - 1

            outputs, _ = self.PaliGemma.llm(
                [None, student_tokens, None],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, student_adarms, None],
            )
            _, student_out, _ = outputs
            v_t = self.action_out_proj_student(student_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt, step_rng

        def cond(carry):
            _, time, _ = carry
            return time >= -dt / 2

        x_0, _, _ = jax.lax.while_loop(cond, step, (noise, 1.0, query_noise_rng))
        return x_0
