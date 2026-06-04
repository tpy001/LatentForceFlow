import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from transformers import FlaxResNetModel


from openpi.models import gemma as _gemma
from openpi.models import model_tavla as _model
from openpi.models import pi0_config
from openpi.models import siglip as _siglip
from openpi.models.pi0_tavla import make_attn_mask, posemb_sincos
from openpi.shared import array_typing as at


_RESNET_ENCODER_REGISTRY = {}


def preload_resnet_encoder(model_name: str) -> None:
    if model_name in _RESNET_ENCODER_REGISTRY:
        return
    model = FlaxResNetModel.from_pretrained(model_name, dtype=jnp.float32)
    _RESNET_ENCODER_REGISTRY[model_name] = (model, model.params)


def _get_resnet_encoder(model_name: str):
    if model_name not in _RESNET_ENCODER_REGISTRY:
        raise ValueError(
            f"ResNet encoder '{model_name}' has not been preloaded. "
            "Call preload_resnet_encoder(...) before initializing training."
        )
    return _RESNET_ENCODER_REGISTRY[model_name]


class Pi0LatentFlowDepthTeachers(_model.BaseModel):
    """Single student expert distilled from separate flow and depth teacher experts."""

    def __init__(self, config: pi0_config.Pi0LatentFlowDepthTeachersConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.effort_type = config.effort_type
        self.distill_layer_indices = tuple(int(i) for i in config.distill_layer_indices)
        self.student_action_loss_weight = float(config.student_action_loss_weight)
        self.flow_teacher_action_loss_weight = float(config.flow_teacher_action_loss_weight)
        self.depth_teacher_action_loss_weight = float(config.depth_teacher_action_loss_weight)
        self.future_flow_align_loss_weight = float(config.future_flow_align_loss_weight)
        self.future_depth_align_loss_weight = float(config.future_depth_align_loss_weight)
        self.flow_token_count = int(config.flow_token_count)
        self.depth_token_count = int(config.depth_token_count)
        self.future_visual_channels = 3
        self.visual_encoder_name = config.visual_encoder_name
        self.qformer_layer_count = int(config.qformer_layer_count)
        self.qformer_mlp_dim = int(config.qformer_mlp_dim)
        self.student_future_query_noise_scale_max = float(config.student_future_query_noise_scale_max)
        self.student_future_query_noise_start_ratio = float(config.student_future_query_noise_start_ratio)
        self.student_future_query_noise_end_ratio = float(config.student_future_query_noise_end_ratio)
        self.uses_train_progress = True

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        student_config = _gemma.get_config(config.action_expert_variant)
        flow_teacher_config = _gemma.get_config(config.flow_teacher_expert_variant)
        depth_teacher_config = _gemma.get_config(config.depth_teacher_expert_variant)
        self.student_width = int(student_config.width)
        self.flow_teacher_width = int(flow_teacher_config.width)
        self.depth_teacher_width = int(depth_teacher_config.width)
        self.flow_projector_hidden_dim = int(
            config.distill_projector_hidden_dim
            if config.distill_projector_hidden_dim is not None
            else flow_teacher_config.width
        )
        self.depth_projector_hidden_dim = int(
            config.depth_distill_projector_hidden_dim
            if config.depth_distill_projector_hidden_dim is not None
            else depth_teacher_config.width
        )

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, student_config, flow_teacher_config, depth_teacher_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True, True, True] if config.pi05 else [False, False, False, False],
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

        self._init_action_path("student", student_config.width, rngs)
        self._init_action_path("flow_teacher", flow_teacher_config.width, rngs)
        self._init_action_path("depth_teacher", depth_teacher_config.width, rngs)

        self.student_future_flow_query = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.flow_token_count, self.student_width), dtype=jnp.float32)
        )
        self.student_future_depth_query = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.depth_token_count, self.student_width), dtype=jnp.float32)
        )

        self._init_visual_teacher("flow", self.flow_token_count, self.flow_teacher_width, config, rngs)
        self._init_visual_teacher("depth", self.depth_token_count, self.depth_teacher_width, config, rngs)

        for layer_ordinal, _ in enumerate(self.distill_layer_indices):
            setattr(
                self,
                f"flow_distill_proj_in_{layer_ordinal}",
                nnx.Linear(self.student_width, self.flow_projector_hidden_dim, rngs=rngs),
            )
            setattr(
                self,
                f"flow_distill_proj_out_{layer_ordinal}",
                nnx.Linear(self.flow_projector_hidden_dim, self.flow_teacher_width, rngs=rngs),
            )
            setattr(
                self,
                f"depth_distill_proj_in_{layer_ordinal}",
                nnx.Linear(self.student_width, self.depth_projector_hidden_dim, rngs=rngs),
            )
            setattr(
                self,
                f"depth_distill_proj_out_{layer_ordinal}",
                nnx.Linear(self.depth_projector_hidden_dim, self.depth_teacher_width, rngs=rngs),
            )

    def _init_action_path(self, name: str, width: int, rngs: nnx.Rngs) -> None:
        setattr(self, f"state_proj_{name}", nnx.Linear(self.action_dim, width, rngs=rngs))
        setattr(self, f"action_in_proj_{name}", nnx.Linear(self.action_dim, width, rngs=rngs))
        setattr(self, f"action_out_proj_{name}", nnx.Linear(width, self.action_dim, rngs=rngs))
        if self.pi05:
            setattr(self, f"{name}_time_mlp_in", nnx.Linear(width, width, rngs=rngs))
            setattr(self, f"{name}_time_mlp_out", nnx.Linear(width, width, rngs=rngs))
        else:
            setattr(self, f"{name}_time_mlp_in", nnx.Linear(2 * width, width, rngs=rngs))
            setattr(self, f"{name}_time_mlp_out", nnx.Linear(width, width, rngs=rngs))

    def _init_visual_teacher(
        self,
        name: str,
        token_count: int,
        width: int,
        config: pi0_config.Pi0LatentFlowDepthTeachersConfig,
        rngs: nnx.Rngs,
    ) -> None:
        resnet, resnet_params = _get_resnet_encoder(config.visual_encoder_name)
        setattr(self, f"{name}_resnet", resnet)
        setattr(self, f"{name}_resnet_params", nnx.Param(jax.tree.map(lambda x: jnp.asarray(x), resnet_params)))
        setattr(self, f"{name}_visual_proj", nnx.Linear(2048, width, rngs=rngs))
        setattr(self, f"{name}_visual_norm", nnx.LayerNorm(num_features=width, rngs=rngs))
        for layer_idx in range(self.qformer_layer_count):
            setattr(self, f"{name}_qformer_query_norm_{layer_idx}", nnx.LayerNorm(num_features=width, rngs=rngs))
            setattr(self, f"{name}_qformer_context_norm_{layer_idx}", nnx.LayerNorm(num_features=width, rngs=rngs))
            setattr(self, f"{name}_qformer_q_proj_{layer_idx}", nnx.Linear(width, width, rngs=rngs))
            setattr(self, f"{name}_qformer_k_proj_{layer_idx}", nnx.Linear(width, width, rngs=rngs))
            setattr(self, f"{name}_qformer_v_proj_{layer_idx}", nnx.Linear(width, width, rngs=rngs))
            setattr(self, f"{name}_qformer_out_proj_{layer_idx}", nnx.Linear(width, width, rngs=rngs))
            setattr(self, f"{name}_qformer_mlp_norm_{layer_idx}", nnx.LayerNorm(num_features=width, rngs=rngs))
            setattr(self, f"{name}_qformer_mlp_in_{layer_idx}", nnx.Linear(width, self.qformer_mlp_dim, rngs=rngs))
            setattr(self, f"{name}_qformer_mlp_out_{layer_idx}", nnx.Linear(self.qformer_mlp_dim, width, rngs=rngs))
        setattr(
            self,
            f"{name}_teacher_future_query",
            nnx.Param(0.02 * jax.random.normal(rngs.params(), (token_count, width), dtype=jnp.float32)),
        )
        setattr(
            self,
            f"{name}_token_embedding",
            nnx.Param(0.02 * jax.random.normal(rngs.params(), (token_count, width), dtype=jnp.float32)),
        )

    def _student_query_noise_scale(self, train_progress: at.Float[at.Array, ""] | float | None):
        if train_progress is None:
            return jnp.asarray(0.0, dtype=jnp.float32)
        progress = jnp.clip(jnp.asarray(train_progress, dtype=jnp.float32), 0.0, 1.0)
        start = jnp.asarray(self.student_future_query_noise_start_ratio, dtype=jnp.float32)
        end = jnp.asarray(self.student_future_query_noise_end_ratio, dtype=jnp.float32)
        max_scale = jnp.asarray(self.student_future_query_noise_scale_max, dtype=jnp.float32)
        ramp = (progress - start) / jnp.maximum(end - start, 1e-6)
        return max_scale * jnp.clip(ramp, 0.0, 1.0)

    def _student_query_tokens(self, batch_size, dtype, *, train, noise_rng, train_progress, query_noise_scale=None):
        flow = jnp.asarray(self.student_future_flow_query.value, dtype=dtype)
        depth = jnp.asarray(self.student_future_depth_query.value, dtype=dtype)
        flow = jnp.broadcast_to(flow[None], (batch_size, *flow.shape))
        depth = jnp.broadcast_to(depth[None], (batch_size, *depth.shape))
        noise_scale_f32 = (
            self._student_query_noise_scale(train_progress)
            if query_noise_scale is None
            else jnp.asarray(query_noise_scale, dtype=jnp.float32)
        )
        noise_scale_f32 = jnp.maximum(noise_scale_f32, 0.0)
        if (not train and query_noise_scale is None) or (
            query_noise_scale is None and float(self.student_future_query_noise_scale_max) <= 0.0
        ):
            return flow, depth, jnp.zeros((batch_size,), dtype=jnp.float32)
        if noise_rng is None:
            raise ValueError("noise_rng is required when student future query noise is enabled.")
        flow_rng, depth_rng = jax.random.split(noise_rng)
        noise_scale = noise_scale_f32.astype(dtype)
        flow_rms = jnp.sqrt(jnp.mean(jnp.square(flow.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6)
        depth_rms = jnp.sqrt(jnp.mean(jnp.square(depth.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6)
        flow = flow + noise_scale * flow_rms.astype(dtype) * jax.random.normal(flow_rng, flow.shape, dtype=dtype)
        depth = depth + noise_scale * depth_rms.astype(dtype) * jax.random.normal(depth_rng, depth.shape, dtype=dtype)
        rate = jnp.ones((batch_size,), dtype=jnp.float32) * jnp.where(noise_scale_f32 > 0.0, 1.0, 0.0)
        return flow, depth, rate

    @staticmethod
    def _require_image(image, field_name: str):
        if image is None:
            raise ValueError(f"Pi0LatentFlowDepthTeachers requires `{field_name}`.")
        return image

    def _visual_images(self, obs: _model.Observation, name: str):
        if name == "flow":
            return (
                self._require_image(obs.flow_img, "observation.flow_img"),
                self._require_image(obs.wrist_flow_img, "observation.wrist_flow_img"),
            )
        if name == "depth":
            return (
                self._require_image(obs.depth_img, "observation.depth_img"),
                self._require_image(obs.wrist_depth_img, "observation.wrist_depth_img"),
            )
        raise ValueError(f"Unknown visual teacher: {name}")

    def _encode_visual(self, image, name: str):
        x = jnp.asarray(image, dtype=jnp.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected {name} image with shape [B, H, W, C], got {x.shape}.")
        if x.shape[-1] != self.future_visual_channels:
            raise ValueError(f"Expected {name} image with {self.future_visual_channels} channels, got {x.shape}.")
        x = jnp.transpose(x, (0, 3, 1, 2))
        outputs = getattr(self, f"{name}_resnet")(
            x,
            params=getattr(self, f"{name}_resnet_params").value,
            train=False,
        )
        latent = einops.rearrange(outputs.last_hidden_state, "b c h w -> b (h w) c")
        latent = getattr(self, f"{name}_visual_proj")(latent)
        return getattr(self, f"{name}_visual_norm")(latent)

    def _compress_visuals(self, obs: _model.Observation, name: str):
        latent = jnp.concatenate([self._encode_visual(image, name) for image in self._visual_images(obs, name)], axis=1)
        query = jnp.asarray(getattr(self, f"{name}_teacher_future_query").value, dtype=latent.dtype)
        query = query + jnp.asarray(getattr(self, f"{name}_token_embedding").value, dtype=latent.dtype)
        query = jnp.broadcast_to(query[None], (latent.shape[0], *query.shape))
        for layer_idx in range(self.qformer_layer_count):
            q = getattr(self, f"{name}_qformer_query_norm_{layer_idx}")(query)
            context = getattr(self, f"{name}_qformer_context_norm_{layer_idx}")(latent)
            q = getattr(self, f"{name}_qformer_q_proj_{layer_idx}")(q)
            keys = getattr(self, f"{name}_qformer_k_proj_{layer_idx}")(context)
            values = getattr(self, f"{name}_qformer_v_proj_{layer_idx}")(context)
            logits = jnp.einsum("bqd,bkd->bqk", q, keys) / jnp.sqrt(jnp.asarray(q.shape[-1], dtype=latent.dtype))
            attn = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(values.dtype)
            attended = jnp.einsum("bqk,bkd->bqd", attn, values)
            query = query + getattr(self, f"{name}_qformer_out_proj_{layer_idx}")(attended)
            mlp = getattr(self, f"{name}_qformer_mlp_norm_{layer_idx}")(query)
            mlp = getattr(self, f"{name}_qformer_mlp_in_{layer_idx}")(mlp)
            mlp = nnx.swish(mlp)
            query = query + getattr(self, f"{name}_qformer_mlp_out_{layer_idx}")(mlp)
        return query

    def embed_prefix(self, obs: _model.Observation):
        tokens = []
        input_mask = []
        ar_mask = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]
        if obs.tokenized_prompt is not None:
            prompt_tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(prompt_tokens)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * prompt_tokens.shape[1]
        return jnp.concatenate(tokens, axis=1), jnp.concatenate(input_mask, axis=1), jnp.array(ar_mask)

    def _embed_action_tokens(self, noisy_actions, timestep, name: str):
        action_tokens = getattr(self, f"action_in_proj_{name}")(noisy_actions)
        width = getattr(self, f"action_in_proj_{name}").out_features
        time_mlp_in = getattr(self, f"{name}_time_mlp_in")
        time_mlp_out = getattr(self, f"{name}_time_mlp_out")
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

    def embed_student_suffix(self, obs, noisy_actions, timestep, *, train, noise_rng, train_progress, query_noise_scale=None):
        flow_query, depth_query, noised_rate = self._student_query_tokens(
            obs.state.shape[0],
            obs.state.dtype,
            train=train,
            noise_rng=noise_rng,
            train_progress=train_progress,
            query_noise_scale=query_noise_scale,
        )
        state = self.state_proj_student(obs.state)[:, None, :]
        actions, adarms = self._embed_action_tokens(noisy_actions, timestep, "student")
        tokens = jnp.concatenate([state, flow_query, depth_query, actions], axis=1)
        ar_mask = jnp.array(
            [False]
            + ([False] * self.flow_token_count)
            + ([False] * self.depth_token_count)
            + [True]
            + ([False] * (actions.shape[1] - 1))
        )
        return tokens, jnp.ones(tokens.shape[:2], dtype=jnp.bool_), ar_mask, adarms, noised_rate

    def embed_teacher_suffix(self, obs, noisy_actions, timestep, name: str):
        teacher = f"{name}_teacher"
        token_count = self.flow_token_count if name == "flow" else self.depth_token_count
        state = getattr(self, f"state_proj_{teacher}")(obs.state)[:, None, :]
        visual = self._compress_visuals(obs, name)
        actions, adarms = self._embed_action_tokens(noisy_actions, timestep, teacher)
        tokens = jnp.concatenate([state, visual, actions], axis=1)
        ar_mask = jnp.array([False] + ([False] * token_count) + [True] + ([False] * (actions.shape[1] - 1)))
        return tokens, jnp.ones(tokens.shape[:2], dtype=jnp.bool_), ar_mask, adarms

    def _forward_all_streams(
        self,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        student_tokens,
        student_mask,
        student_ar_mask,
        student_adarms,
        flow_tokens,
        flow_mask,
        flow_ar_mask,
        flow_adarms,
        depth_tokens,
        depth_mask,
        depth_ar_mask,
        depth_adarms,
    ):
        bsz = prefix_mask.shape[0]
        p_len = prefix_mask.shape[1]
        student_len = student_mask.shape[1]
        flow_len = flow_mask.shape[1]
        depth_len = depth_mask.shape[1]

        prefix_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
        student_attn = make_attn_mask(student_mask, student_ar_mask)
        flow_attn = make_attn_mask(flow_mask, flow_ar_mask)
        depth_attn = make_attn_mask(depth_mask, depth_ar_mask)

        def suffix_to_prefix(suffix_mask, allow_prefix):
            attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_mask.shape[1])
            attn = jnp.logical_and(attn, suffix_mask[:, :, None])
            return jnp.logical_and(attn, allow_prefix[None, :, None])

        student_prefix = suffix_to_prefix(
            student_mask,
            jnp.array(
                [True]
                + ([True] * self.flow_token_count)
                + ([True] * self.depth_token_count)
                + ([True] * self.action_horizon)
            ),
        )
        flow_prefix = suffix_to_prefix(
            flow_mask,
            jnp.array([True] + ([True] * self.flow_token_count) + ([True] * self.action_horizon)),
        )
        depth_prefix = suffix_to_prefix(
            depth_mask,
            jnp.array([True] + ([True] * self.depth_token_count) + ([True] * self.action_horizon)),
        )

        prefix_row = jnp.concatenate(
            [
                prefix_attn,
                jnp.zeros((bsz, p_len, student_len), dtype=jnp.bool_),
                jnp.zeros((bsz, p_len, flow_len), dtype=jnp.bool_),
                jnp.zeros((bsz, p_len, depth_len), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        student_row = jnp.concatenate(
            [
                student_prefix,
                student_attn,
                jnp.zeros((bsz, student_len, flow_len), dtype=jnp.bool_),
                jnp.zeros((bsz, student_len, depth_len), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        flow_row = jnp.concatenate(
            [
                flow_prefix,
                jnp.zeros((bsz, flow_len, student_len), dtype=jnp.bool_),
                flow_attn,
                jnp.zeros((bsz, flow_len, depth_len), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        depth_row = jnp.concatenate(
            [
                depth_prefix,
                jnp.zeros((bsz, depth_len, student_len), dtype=jnp.bool_),
                jnp.zeros((bsz, depth_len, flow_len), dtype=jnp.bool_),
                depth_attn,
            ],
            axis=-1,
        )
        full_attn = jnp.concatenate([prefix_row, student_row, flow_row, depth_row], axis=1)

        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_len = jnp.sum(prefix_mask, axis=-1)[:, None]
        student_positions = prefix_len + jnp.cumsum(student_mask, axis=-1) - 1
        flow_positions = prefix_len + jnp.cumsum(flow_mask, axis=-1) - 1
        depth_positions = prefix_len + jnp.cumsum(depth_mask, axis=-1) - 1
        positions = jnp.concatenate([prefix_positions, student_positions, flow_positions, depth_positions], axis=1)
        (outputs, layers), _ = self.PaliGemma.llm(
            [prefix_tokens, student_tokens, flow_tokens, depth_tokens],
            mask=full_attn,
            positions=positions,
            adarms_cond=[None, student_adarms, flow_adarms, depth_adarms],
            return_layer_indices=self.distill_layer_indices,
        )
        return outputs, layers

    @staticmethod
    def _cosine_distance_masked(lhs, rhs, token_mask):
        lhs = lhs.astype(jnp.float32)
        rhs = rhs.astype(jnp.float32)
        lhs_norm = lhs / jnp.sqrt(jnp.sum(jnp.square(lhs), axis=-1, keepdims=True) + 1e-6)
        rhs_norm = rhs / jnp.sqrt(jnp.sum(jnp.square(rhs), axis=-1, keepdims=True) + 1e-6)
        losses = 1.0 - jnp.sum(lhs_norm * rhs_norm, axis=-1)
        weights = token_mask.astype(losses.dtype)
        denom = jnp.maximum(jnp.sum(weights, axis=-1), jnp.asarray(1.0, dtype=losses.dtype))
        return jnp.sum(losses * weights, axis=-1) / denom

    def _project_flow_distill(self, hidden, layer_ordinal: int):
        hidden = getattr(self, f"flow_distill_proj_in_{layer_ordinal}")(hidden)
        hidden = nnx.swish(hidden)
        return getattr(self, f"flow_distill_proj_out_{layer_ordinal}")(hidden)

    def _project_depth_distill(self, hidden, layer_ordinal: int):
        hidden = getattr(self, f"depth_distill_proj_in_{layer_ordinal}")(hidden)
        hidden = nnx.swish(hidden)
        return getattr(self, f"depth_distill_proj_out_{layer_ordinal}")(hidden)

    @staticmethod
    def _restore_aux_images(processed, original):
        updates = {}
        for key in ("flow_img", "wrist_flow_img", "depth_img", "wrist_depth_img"):
            value = getattr(original, key)
            if value is not None:
                updates[key] = value
        return processed if not updates else processed.replace(**updates)

    def compute_loss_with_stats(self, rng, observation, actions, *, train=False, train_progress=None):
        preprocess_rng, noise_rng, time_rng, query_noise_rng = jax.random.split(rng, 4)
        original_observation = observation
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train, effort_type=self.effort_type)
        observation = self._restore_aux_images(observation, original_observation)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t_action = time_expanded * noise + (1 - time_expanded) * actions
        u_t_action = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        student_tokens, student_mask, student_ar_mask, student_adarms, noised_rate = self.embed_student_suffix(
            observation,
            x_t_action,
            time,
            train=train,
            noise_rng=query_noise_rng,
            train_progress=train_progress,
        )
        flow_tokens, flow_mask, flow_ar_mask, flow_adarms = self.embed_teacher_suffix(observation, x_t_action, time, "flow")
        depth_tokens, depth_mask, depth_ar_mask, depth_adarms = self.embed_teacher_suffix(
            observation, x_t_action, time, "depth"
        )

        outputs, layers = self._forward_all_streams(
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            student_tokens,
            student_mask,
            student_ar_mask,
            student_adarms,
            flow_tokens,
            flow_mask,
            flow_ar_mask,
            flow_adarms,
            depth_tokens,
            depth_mask,
            depth_ar_mask,
            depth_adarms,
        )
        _, student_out, flow_out, depth_out = outputs

        student_v = self.action_out_proj_student(student_out[:, -self.action_horizon :])
        flow_v = self.action_out_proj_flow_teacher(flow_out[:, -self.action_horizon :])
        depth_v = self.action_out_proj_depth_teacher(depth_out[:, -self.action_horizon :])
        student_action_loss = jnp.mean(jnp.square(student_v - u_t_action), axis=(-2, -1))
        flow_teacher_action_loss = jnp.mean(jnp.square(flow_v - u_t_action), axis=(-2, -1))
        depth_teacher_action_loss = jnp.mean(jnp.square(depth_v - u_t_action), axis=(-2, -1))

        flow_slice = slice(1, 1 + self.flow_token_count)
        depth_slice = slice(1 + self.flow_token_count, 1 + self.flow_token_count + self.depth_token_count)
        teacher_slice = lambda count: slice(1, 1 + count)
        flow_mask_tokens = jnp.ones((actions.shape[0], self.flow_token_count), dtype=jnp.bool_)
        depth_mask_tokens = jnp.ones((actions.shape[0], self.depth_token_count), dtype=jnp.bool_)
        flow_losses = []
        depth_losses = []
        for layer_ordinal, layer in enumerate(layers):
            _, student_hidden, flow_hidden, depth_hidden = layer
            flow_losses.append(
                self._cosine_distance_masked(
                    self._project_flow_distill(student_hidden[:, flow_slice, :], layer_ordinal),
                    jax.lax.stop_gradient(flow_hidden[:, teacher_slice(self.flow_token_count), :]),
                    flow_mask_tokens,
                )
            )
            depth_losses.append(
                self._cosine_distance_masked(
                    self._project_depth_distill(student_hidden[:, depth_slice, :], layer_ordinal),
                    jax.lax.stop_gradient(depth_hidden[:, teacher_slice(self.depth_token_count), :]),
                    depth_mask_tokens,
                )
            )

        future_flow_align_loss = jnp.mean(jnp.stack(flow_losses, axis=0), axis=0)
        future_depth_align_loss = jnp.mean(jnp.stack(depth_losses, axis=0), axis=0)
        total_loss = (
            self.student_action_loss_weight * student_action_loss
            + self.flow_teacher_action_loss_weight * flow_teacher_action_loss
            + self.depth_teacher_action_loss_weight * depth_teacher_action_loss
            + self.future_flow_align_loss_weight * future_flow_align_loss
            + self.future_depth_align_loss_weight * future_depth_align_loss
        )
        stats = {
            "loss/student_action": student_action_loss,
            "loss/flow_teacher_action": flow_teacher_action_loss,
            "loss/depth_teacher_action": depth_teacher_action_loss,
            "loss/distill_future_flow": future_flow_align_loss,
            "loss/distill_future_depth": future_depth_align_loss,
            "noise/student_future_query_token_rate": jnp.mean(noised_rate),
            "noise/student_future_query_scale": self._student_query_noise_scale(train_progress),
            "loss/total": total_loss,
        }
        return total_loss, stats

    def compute_loss(self, rng, observation, actions, *, train=False, train_progress=None):
        loss, _ = self.compute_loss_with_stats(rng, observation, actions, train=train, train_progress=train_progress)
        return loss

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None, debug_query_noise_scale=None):
        original_observation = observation
        observation = _model.preprocess_observation(None, observation, train=False, effort_type=self.effort_type)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        action_noise_rng, query_noise_rng = jax.random.split(rng)
        if noise is None:
            noise = jax.random.normal(action_noise_rng, (batch_size, self.action_horizon, self.action_dim))
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            adarms_cond=[None, None, None, None],
        )

        def step(carry):
            x_t, time, step_rng = carry
            step_rng, iter_query_noise_rng = jax.random.split(step_rng)
            student_tokens, student_mask, student_ar_mask, student_adarms, _ = self.embed_student_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
                train=False,
                noise_rng=iter_query_noise_rng,
                train_progress=None,
                query_noise_scale=debug_query_noise_scale,
            )
            student_attn_mask = make_attn_mask(student_mask, student_ar_mask)
            prefix_to_student = einops.repeat(prefix_mask, "b p -> b s p", s=student_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_to_student, student_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(student_mask, axis=-1) - 1
            outputs, _ = self.PaliGemma.llm(
                [None, student_tokens, None, None],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, student_adarms, None, None],
            )
            _, student_out, _, _ = outputs
            v_t = self.action_out_proj_student(student_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt, step_rng

        def cond(carry):
            _, time, _ = carry
            return time >= -dt / 2

        x_0, _, _ = jax.lax.while_loop(cond, step, (noise, 1.0, query_noise_rng))
        return x_0
