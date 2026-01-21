"""Segment-id attention mask patch for megalodon_jax.

This provides temporary support for block-diagonal attention masks derived from
segment IDs during training. It is intended as a minimal shim until upstream
megalodon_jax adds first-class segment-id masking support.
"""

from __future__ import annotations

from typing import Any


def apply_segment_ids_patch() -> bool:
    """Monkeypatch megalodon_jax to accept segment_ids for block-diagonal attention.

    :return bool: True if the patch is applied or already present, False if unavailable.
    """

    try:
        import inspect

        import equinox as eqx
        import jax
        import jax.numpy as jnp
        from megalodon_jax import model as model_mod
        from megalodon_jax.layers import attention as attn_mod
        from megalodon_jax.layers.attention import attention_multi_chunk as _orig_multi
        from megalodon_jax.layers.attention import attention_single_chunk as _orig_single
        from megalodon_jax.ops import matmul_3d_weight
        from megalodon_jax.types import EMAState, LayerCache, ModelCache
    except Exception:  # pragma: no cover - optional dependency
        return False

    if "segment_ids" in inspect.signature(model_mod.MegalodonForCausalLM.compute_loss).parameters:
        return True

    if getattr(attn_mod, "_CHOMP_SEGMENT_IDS_PATCHED", False):
        return True

    def _apply_segment_mask(scores: jax.Array, segment_ids: jax.Array) -> jax.Array:
        """Apply a block-diagonal mask based on segment IDs.

        :param jax.Array scores: Attention scores of shape [B, H, L, L].
        :param jax.Array segment_ids: Segment IDs of shape [B, L].
        :return jax.Array: Masked scores.
        """
        seg = segment_ids.astype(jnp.int32)
        valid = seg > 0
        same = seg[:, :, None] == seg[:, None, :]
        same = same & valid[:, :, None] & valid[:, None, :]
        same = same[:, None, :, :]
        return jnp.where(same, scores, -jnp.inf)

    def attention_single_chunk(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        kv_mask: jax.Array | None = None,
        accum_dtype: jnp.dtype = jnp.float32,
        causal: bool = True,
        dropout_rate: float = 0.0,
        deterministic: bool = True,
        key: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
    ) -> jax.Array:
        """Single-chunk attention with optional segment mask.

        :param jax.Array q: Query tensor [B, L, H, Dh].
        :param jax.Array k: Key tensor [B, L, H, Dh].
        :param jax.Array v: Value tensor [B, L, H, Dv].
        :param kv_mask: Optional key/value mask [B, L].
        :param jnp.dtype accum_dtype: Accumulation dtype for matmul.
        :param bool causal: Whether to apply causal masking.
        :param float dropout_rate: Dropout rate for attention weights.
        :param bool deterministic: If False, apply dropout.
        :param key: PRNG key when dropout is enabled.
        :param segment_ids: Optional segment IDs [B, L].
        :return jax.Array: Output tensor [B, L, H, Dv].
        """
        if segment_ids is None:
            return _orig_single(
                q,
                k,
                v,
                kv_mask=kv_mask,
                accum_dtype=accum_dtype,
                causal=causal,
                dropout_rate=dropout_rate,
                deterministic=deterministic,
                key=key,
            )

        B, L_q, H, Dh = q.shape
        L_kv = k.shape[1]
        Dv = v.shape[-1]

        if L_q == 0 or L_kv == 0:
            return jnp.zeros((B, L_q, H, Dv), dtype=q.dtype)

        if segment_ids.shape != (B, L_kv) or L_q != L_kv:
            raise ValueError("segment_ids requires shape (batch, seq) with L_q == L_kv")

        scores = jnp.einsum(
            "bqhd,bkhd->bhqk",
            q,
            k,
            preferred_element_type=accum_dtype,
        )

        neg_inf = -jnp.inf
        if causal and L_q == L_kv:
            causal_mask = jnp.tril(jnp.ones((L_q, L_kv), dtype=jnp.bool_))
            scores = jnp.where(causal_mask, scores, neg_inf)
        elif causal and L_q < L_kv:
            q_pos = jnp.arange(L_q)[:, None]
            k_pos = jnp.arange(L_kv)[None, :]
            offset = L_kv - L_q
            causal_mask = k_pos <= (q_pos + offset)
            scores = jnp.where(causal_mask, scores, neg_inf)

        scores = _apply_segment_mask(scores, segment_ids)

        if kv_mask is not None:
            kv_mask_expanded = kv_mask[:, None, None, :]
            scores = jnp.where(kv_mask_expanded, scores, neg_inf)

        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_weights = jnp.where(jnp.isnan(attn_weights), 0.0, attn_weights)

        if not deterministic and dropout_rate > 0.0:
            if key is None:
                raise ValueError("PRNG key required for dropout")
            keep_mask = jax.random.bernoulli(key, 1.0 - dropout_rate, attn_weights.shape)
            inv_keep = jnp.asarray(1.0 / (1.0 - dropout_rate), dtype=attn_weights.dtype)
            attn_weights = attn_weights * keep_mask.astype(attn_weights.dtype) * inv_keep

        out = jnp.einsum(
            "bhqk,bkhd->bqhd",
            attn_weights,
            v,
            preferred_element_type=accum_dtype,
        )

        return out.astype(q.dtype)

    def attention_multi_chunk(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        chunk_size: int,
        start_index: jax.Array,
        rotary: Any,
        mask: jax.Array | None = None,
        accum_dtype: jnp.dtype = jnp.float32,
        dropout_rate: float = 0.0,
        deterministic: bool = True,
        key: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
    ) -> jax.Array:
        """Chunked attention with optional segment mask.

        :param jax.Array q: Query tensor [B, L, H, Dh].
        :param jax.Array k: Key tensor [B, L, H, Dh].
        :param jax.Array v: Value tensor [B, L, H, Dv].
        :param int chunk_size: Chunk length for attention.
        :param jax.Array start_index: Rotary start index.
        :param rotary: Rotary embedding callable.
        :param mask: Optional padding mask [B, L].
        :param jnp.dtype accum_dtype: Accumulation dtype for matmul.
        :param float dropout_rate: Dropout rate for attention weights.
        :param bool deterministic: If False, apply dropout.
        :param key: PRNG key when dropout is enabled.
        :param segment_ids: Optional segment IDs [B, L].
        :return jax.Array: Output tensor [B, L, H, Dv].
        """
        if segment_ids is None:
            return _orig_multi(
                q,
                k,
                v,
                chunk_size=chunk_size,
                start_index=start_index,
                rotary=rotary,
                mask=mask,
                accum_dtype=accum_dtype,
                dropout_rate=dropout_rate,
                deterministic=deterministic,
                key=key,
            )

        B, L, H, Dh = q.shape
        Dv = v.shape[-1]

        if L == 0:
            return jnp.zeros((B, L, H, Dv), dtype=q.dtype)

        if segment_ids.shape != (B, L):
            raise ValueError("segment_ids must have shape (batch, seq)")

        if mask is not None:
            segment_ids = jnp.where(mask, segment_ids, 0)

        if chunk_size >= L:
            q_rot, k_rot = rotary(q, k, start_index)
            return attention_single_chunk(
                q_rot,
                k_rot,
                v,
                kv_mask=mask,
                accum_dtype=accum_dtype,
                causal=True,
                dropout_rate=dropout_rate,
                deterministic=deterministic,
                key=key,
                segment_ids=segment_ids,
            )

        pad_len = (chunk_size - L % chunk_size) % chunk_size
        if pad_len > 0:
            q = jnp.pad(q, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            k = jnp.pad(k, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            v = jnp.pad(v, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            if mask is not None:
                mask = jnp.pad(mask, ((0, 0), (0, pad_len)), constant_values=False)
            segment_ids = jnp.pad(segment_ids, ((0, 0), (0, pad_len)), constant_values=0)

        L_padded = q.shape[1]
        num_chunks = L_padded // chunk_size

        q_rot_full, k_rot_full = rotary(q, k, start_index)
        q_rot = q_rot_full.reshape(B * num_chunks, chunk_size, H, Dh)
        k_rot = k_rot_full.reshape(B * num_chunks, chunk_size, H, Dh)
        v_chunked = v.reshape(B, num_chunks, chunk_size, H, Dv).reshape(
            B * num_chunks, chunk_size, H, Dv
        )

        mask_chunked = None
        if mask is not None:
            mask_chunked = mask.reshape(B * num_chunks, chunk_size)

        seg_chunked = segment_ids.reshape(B * num_chunks, chunk_size)

        out_chunked = attention_single_chunk(
            q_rot,
            k_rot,
            v_chunked,
            kv_mask=mask_chunked,
            accum_dtype=accum_dtype,
            dropout_rate=dropout_rate,
            deterministic=deterministic,
            key=key,
            segment_ids=seg_chunked,
        )

        out = out_chunked.reshape(B, L_padded, H, Dv)
        if pad_len > 0:
            out = out[:, :L, :, :]
        return out

    orig_chunked_call = attn_mod.ChunkedAttention.__call__

    def chunked_call(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        cache: Any | None = None,
        mask: jax.Array | None = None,
        return_cache: bool = False,
        deterministic: bool = True,
        key: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
    ) -> tuple[jax.Array, Any | None, jax.Array]:
        """Patched ChunkedAttention call supporting segment IDs.

        :param jax.Array q: Query tensor [B, L, H, Dh].
        :param jax.Array k: Key tensor [B, L, H, Dh].
        :param jax.Array v: Value tensor [B, L, H, Dv].
        :param cache: Optional attention cache.
        :param mask: Optional padding mask [B, L].
        :param bool return_cache: Whether to return cache.
        :param bool deterministic: If False, apply dropout.
        :param key: PRNG key when dropout is enabled.
        :param segment_ids: Optional segment IDs [B, L].
        :return tuple: (output, cache, position) tuple.
        """
        if segment_ids is None:
            return orig_chunked_call(
                self,
                q,
                k,
                v,
                cache=cache,
                mask=mask,
                return_cache=return_cache,
                deterministic=deterministic,
                key=key,
            )
        if cache is not None or return_cache:
            raise ValueError("segment_ids are not supported with cached attention")
        out = attention_multi_chunk(
            q,
            k,
            v,
            chunk_size=self.chunk_size,
            start_index=jnp.array(0, dtype=jnp.int32),
            rotary=self.rotary,
            mask=mask,
            accum_dtype=self.accum_dtype,
            dropout_rate=self.attention_dropout,
            deterministic=deterministic,
            key=key,
            segment_ids=segment_ids,
        )
        return out, None, jnp.array(q.shape[1], dtype=jnp.int32)

    orig_attn_call = attn_mod.MegalodonAttention.__call__

    def attn_call(
        self,
        x: jax.Array,
        cache: LayerCache | None = None,
        mask: jax.Array | None = None,
        return_cache: bool = False,
        deterministic: bool = True,
        key: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
    ) -> tuple[jax.Array, LayerCache | None]:
        """Patched MegalodonAttention call supporting segment IDs.

        :param jax.Array x: Input tensor [B, L, D].
        :param cache: Optional layer cache.
        :param mask: Optional padding mask [B, L].
        :param bool return_cache: Whether to return cache.
        :param bool deterministic: If False, apply dropout.
        :param key: PRNG key when dropout is enabled.
        :param segment_ids: Optional segment IDs [B, L].
        :return tuple: (output, new_cache) tuple.
        """
        if segment_ids is None:
            return orig_attn_call(
                self,
                x,
                cache=cache,
                mask=mask,
                return_cache=return_cache,
                deterministic=deterministic,
                key=key,
            )

        B, L, D = x.shape
        H = self.num_heads
        Dh = self.head_dim
        Dv = self.value_head_dim

        x = x.astype(self.compute_dtype)

        if key is not None:
            k1, k2, k3, k4 = jax.random.split(key, 4)
        else:
            k1 = k2 = k3 = k4 = None

        norm_state = cache.norm if cache is not None else None
        ema_state = cache.ema.h if cache is not None and cache.ema is not None else None
        attn_cache = cache.attn if cache is not None else None

        x_tn, new_norm_state = self.timenorm(x, state=norm_state, mask=mask)

        need_ema_state = return_cache or ema_state is not None
        y_cema, h_last = self.cema(
            x_tn.transpose(0, 2, 1),
            h_init=ema_state,
            return_state=need_ema_state,
            mask=mask,
        )
        y_cema = y_cema.transpose(0, 2, 1)

        mx = self.rmsnorm(y_cema)
        if not deterministic and self.hidden_dropout > 0.0 and k2 is not None:
            keep = jax.random.bernoulli(k2, 1.0 - self.hidden_dropout, mx.shape)
            inv_keep = jnp.asarray(1.0 / (1.0 - self.hidden_dropout), dtype=mx.dtype)
            mx = jnp.where(keep, mx * inv_keep, jnp.zeros((), dtype=mx.dtype))

        z = attn_mod.linear_3d(self.wz, mx, self.compute_dtype, self.accum_dtype, self.gemm_backend)
        z = z.reshape(B, L, H, Dh)

        z_f32 = z.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(z_f32**2, axis=-1, keepdims=True) + self.norm_eps)
        z_normed = z_f32 / rms

        gamma_heads = self.gamma.reshape(2, H, Dh).astype(jnp.float32)
        beta_heads = self.beta.reshape(2, H, Dh).astype(jnp.float32)
        scale = (gamma_heads + 1.0) / jnp.sqrt(jnp.asarray(Dh, dtype=jnp.float32))

        q = (z_normed * scale[0] + beta_heads[0]).astype(self.compute_dtype)
        k = (z_normed * scale[1] + beta_heads[1]).astype(self.compute_dtype)

        v = jax.nn.silu(
            attn_mod.linear_3d(
                self.wv, x_tn, self.compute_dtype, self.accum_dtype, self.gemm_backend
            )
        )
        v = v.reshape(B, L, H, Dv)

        r = jax.nn.silu(
            attn_mod.linear_3d(self.wr, mx, self.compute_dtype, self.accum_dtype, self.gemm_backend)
        )

        out, new_attn_cache, new_position = self.inner(
            q,
            k,
            v,
            cache=attn_cache,
            mask=mask,
            return_cache=return_cache,
            deterministic=deterministic,
            key=k1,
            segment_ids=segment_ids,
        )

        out = out.reshape(B, L, self.value_dim)
        gated = out * r

        if not deterministic and self.hidden_dropout > 0.0 and k3 is not None:
            keep = jax.random.bernoulli(k3, 1.0 - self.hidden_dropout, gated.shape)
            inv_keep = jnp.asarray(1.0 / (1.0 - self.hidden_dropout), dtype=gated.dtype)
            gated = jnp.where(keep, gated * inv_keep, jnp.zeros((), dtype=gated.dtype))

        h = attn_mod.linear_3d(
            self.wh1, mx, self.compute_dtype, self.accum_dtype, self.gemm_backend
        ) + attn_mod.linear_3d(
            self.wh2, gated, self.compute_dtype, self.accum_dtype, self.gemm_backend
        )

        if not deterministic and self.dropout > 0.0 and k4 is not None:
            keep = jax.random.bernoulli(k4, 1.0 - self.dropout, h.shape)
            inv_keep = jnp.asarray(1.0 / (1.0 - self.dropout), dtype=h.dtype)
            h = jnp.where(keep, h * inv_keep, jnp.zeros((), dtype=h.dtype))

        y = h + x

        if return_cache:
            new_cache = LayerCache(
                attn=new_attn_cache,
                norm=new_norm_state,
                ema=EMAState(h=h_last) if h_last is not None else None,
                position=new_position
                if new_position is not None
                else jnp.array(0, dtype=jnp.int32),
            )
        else:
            new_cache = None

        return y, new_cache

    @eqx.filter_checkpoint
    def _checkpointed_layer(
        layer: Any,
        x: jax.Array,
        mask: jax.Array | None,
        key: jax.Array | None,
        segment_ids: jax.Array | None = None,
    ) -> jax.Array:
        """Checkpointed layer forward with optional segment IDs.

        :param layer: Layer module to call.
        :param jax.Array x: Input tensor [B, L, D].
        :param mask: Optional padding mask [B, L].
        :param key: PRNG key for dropout.
        :param segment_ids: Optional segment IDs [B, L].
        :return jax.Array: Layer output.
        """
        out, _ = layer(
            x,
            cache=None,
            mask=mask,
            return_cache=False,
            deterministic=False,
            key=key,
            segment_ids=segment_ids,
        )
        return out

    orig_block_call = model_mod.MegalodonBlock.__call__

    def block_call(
        self,
        x: jax.Array,
        cache: LayerCache | None = None,
        mask: jax.Array | None = None,
        return_cache: bool = False,
        deterministic: bool = True,
        key: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
    ) -> tuple[jax.Array, LayerCache | None]:
        """Patched MegalodonBlock call supporting segment IDs.

        :param jax.Array x: Input tensor [B, L, D].
        :param cache: Optional layer cache.
        :param mask: Optional padding mask [B, L].
        :param bool return_cache: Whether to return cache.
        :param bool deterministic: If False, apply dropout.
        :param key: PRNG key when dropout is enabled.
        :param segment_ids: Optional segment IDs [B, L].
        :return tuple: (output, new_cache) tuple.
        """
        if segment_ids is None:
            return orig_block_call(
                self,
                x,
                cache=cache,
                mask=mask,
                return_cache=return_cache,
                deterministic=deterministic,
                key=key,
            )

        residual_base = x
        if key is not None:
            k_attn, k_ffn = jax.random.split(key)
        else:
            k_attn = k_ffn = None

        x, cache = self.attn(
            x,
            cache=cache,
            mask=mask,
            return_cache=return_cache,
            deterministic=deterministic,
            key=k_attn,
            segment_ids=segment_ids,
        )

        x = self.ffn(
            x,
            residual_base=residual_base,
            deterministic=deterministic,
            key=k_ffn,
        )

        return x, cache

    orig_model_call = model_mod.MegalodonModel.__call__

    def model_call(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        cache: ModelCache | None = None,
        return_cache: bool = False,
        deterministic: bool = True,
        key: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
    ) -> tuple[jax.Array, ModelCache | None]:
        """Patched MegalodonModel call supporting segment IDs.

        :param jax.Array input_ids: Input token IDs [B, L].
        :param attention_mask: Optional padding mask [B, L].
        :param cache: Optional model cache.
        :param bool return_cache: Whether to return cache.
        :param bool deterministic: If False, apply dropout.
        :param key: PRNG key when dropout is enabled.
        :param segment_ids: Optional segment IDs [B, L].
        :return tuple: (hidden, cache) tuple.
        """
        if segment_ids is None:
            return orig_model_call(
                self,
                input_ids,
                attention_mask=attention_mask,
                cache=cache,
                return_cache=return_cache,
                deterministic=deterministic,
                key=key,
            )

        if (
            not deterministic
            and key is None
            and (
                self.config.dropout > 0.0
                or self.config.attention_dropout > 0.0
                or self.config.hidden_dropout > 0.0
            )
        ):
            raise ValueError(
                "PRNG key required when deterministic=False and dropout is enabled. "
                "Pass a key via `key=jax.random.PRNGKey(...)` or set deterministic=True."
            )

        B, L = input_ids.shape
        if B == 0 or L == 0:
            empty_hidden = jnp.zeros((B, L, self.config.model_dim), dtype=self.config.compute_dtype)
            if return_cache:
                empty_cache = (
                    cache
                    if cache is not None
                    else ModelCache(tuple([None] * len(self.layers)), None)
                )
            else:
                empty_cache = None
            return empty_hidden, empty_cache

        layer_return_cache = return_cache and deterministic

        vocab_size = self.config.vocab_size
        has_invalid_tokens = jnp.any((input_ids < 0) | (input_ids >= vocab_size))
        input_ids = eqx.error_if(
            input_ids,
            has_invalid_tokens,
            f"input_ids contain out-of-bounds values. Valid range: [0, {vocab_size})",
        )

        x = self.embed.weight[input_ids]
        if self.scale != 1.0:
            x = x * jnp.asarray(self.scale, dtype=x.dtype)

        pad_mask = input_ids == self.config.pad_token_id
        x = jnp.where(pad_mask[:, :, None], jnp.zeros((), dtype=x.dtype), x)

        if x.dtype != self.config.compute_dtype:
            x = x.astype(self.config.compute_dtype)

        uses_streaming = layer_return_cache or cache is not None
        if uses_streaming and attention_mask is not None:
            has_padding = ~jnp.all(attention_mask)
            x = eqx.error_if(
                x,
                has_padding,
                "Cannot use cache with padding in attention_mask. "
                "Caching is only supported for autoregressive generation without padding. "
                "Use cache=None and return_cache=False for padded prefill.",
            )

        if cache is not None:
            if len(cache.layer_caches) != len(self.layers):
                raise ValueError(
                    f"Cache has {len(cache.layer_caches)} layer entries, "
                    f"expected {len(self.layers)}"
                )
            layer_caches = list(cache.layer_caches)
            final_norm_state = cache.final_norm
        else:
            layer_caches = [None] * len(self.layers)
            final_norm_state = None

        if key is not None:
            keys = list(jax.random.split(key, len(self.layers)))
        else:
            keys = [None] * len(self.layers)

        use_ckpt = self.use_checkpoint and not deterministic
        new_caches: list[LayerCache | None] = []
        for layer, layer_cache, layer_key in zip(self.layers, layer_caches, keys, strict=True):
            if use_ckpt:
                x = _checkpointed_layer(
                    layer, x, attention_mask, layer_key, segment_ids=segment_ids
                )
                new_caches.append(None)
            else:
                x, new_cache = layer(
                    x,
                    cache=layer_cache,
                    mask=attention_mask,
                    return_cache=layer_return_cache,
                    deterministic=deterministic,
                    key=layer_key,
                    segment_ids=segment_ids,
                )
                new_caches.append(new_cache)

        x, final_norm_state = self.norm(x, state=final_norm_state, mask=attention_mask)

        if return_cache:
            out_cache = ModelCache(
                layer_caches=tuple(new_caches),
                final_norm=final_norm_state,
            )
            out_cache = jax.tree.map(model_mod._stop_if_array, out_cache)
        else:
            out_cache = None

        return x, out_cache

    orig_lm_call = model_mod.MegalodonForCausalLM.__call__

    def lm_call(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        cache: ModelCache | None = None,
        return_cache: bool = False,
        deterministic: bool = True,
        key: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
    ) -> tuple[jax.Array, ModelCache | None]:
        """Patched MegalodonForCausalLM call supporting segment IDs.

        :param jax.Array input_ids: Input token IDs [B, L].
        :param attention_mask: Optional padding mask [B, L].
        :param cache: Optional model cache.
        :param bool return_cache: Whether to return cache.
        :param bool deterministic: If False, apply dropout.
        :param key: PRNG key when dropout is enabled.
        :param segment_ids: Optional segment IDs [B, L].
        :return tuple: (logits, cache) tuple.
        """
        if segment_ids is None:
            return orig_lm_call(
                self,
                input_ids,
                attention_mask=attention_mask,
                cache=cache,
                return_cache=return_cache,
                deterministic=deterministic,
                key=key,
            )

        if (
            not deterministic
            and key is None
            and (
                self.config.dropout > 0.0
                or self.config.attention_dropout > 0.0
                or self.config.hidden_dropout > 0.0
            )
        ):
            raise ValueError(
                "PRNG key required when deterministic=False and dropout is enabled. "
                "Pass a key via `key=jax.random.PRNGKey(...)` or set deterministic=True."
            )

        hidden, cache = self.model(
            input_ids,
            attention_mask=attention_mask,
            cache=cache,
            return_cache=return_cache,
            deterministic=deterministic,
            key=key,
            segment_ids=segment_ids,
        )

        compute_dtype = self.config.compute_dtype
        accum_dtype = self.config.accum_dtype
        gemm_backend = self.config.gemm_backend
        if self.tied:
            logits = matmul_3d_weight(
                hidden,
                self.model.embed.weight,
                compute_dtype,
                accum_dtype,
                gemm_backend,
            )
        else:
            logits = matmul_3d_weight(
                hidden,
                self.lm_head.weight,
                compute_dtype,
                accum_dtype,
                gemm_backend,
            )

        return logits, cache

    orig_compute_loss = model_mod.MegalodonForCausalLM.compute_loss

    def compute_loss(
        self,
        input_ids: jax.Array,
        labels: jax.Array,
        attention_mask: jax.Array | None = None,
        ignore_index: int = -100,
        deterministic: bool = True,
        key: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
    ) -> jax.Array:
        """Patched loss with optional segment IDs.

        :param jax.Array input_ids: Input token IDs [B, L].
        :param jax.Array labels: Label token IDs [B, L].
        :param attention_mask: Optional padding mask [B, L].
        :param int ignore_index: Label value to ignore in loss.
        :param bool deterministic: If False, apply dropout.
        :param key: PRNG key when dropout is enabled.
        :param segment_ids: Optional segment IDs [B, L].
        :return jax.Array: Scalar loss.
        """
        if segment_ids is None:
            return orig_compute_loss(
                self,
                input_ids,
                labels,
                attention_mask=attention_mask,
                ignore_index=ignore_index,
                deterministic=deterministic,
                key=key,
            )

        if (
            not deterministic
            and key is None
            and (
                self.config.dropout > 0.0
                or self.config.attention_dropout > 0.0
                or self.config.hidden_dropout > 0.0
            )
        ):
            raise ValueError(
                "PRNG key required when deterministic=False and dropout is enabled. "
                "Pass a key via `key=jax.random.PRNGKey(...)` or set deterministic=True."
            )

        logits, _ = self(
            input_ids,
            attention_mask=attention_mask,
            cache=None,
            return_cache=False,
            deterministic=deterministic,
            key=key,
            segment_ids=segment_ids,
        )

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        softmax_dtype = self.config.softmax_dtype
        if shift_labels.shape[1] == 0:
            return jnp.zeros((), dtype=softmax_dtype)

        valid_mask = shift_labels != ignore_index
        if attention_mask is not None:
            shift_attn_mask = attention_mask[:, 1:]
            valid_mask = valid_mask & shift_attn_mask

        vocab_size = shift_logits.shape[-1]
        has_invalid_labels = jnp.any(
            valid_mask & ((shift_labels < 0) | (shift_labels >= vocab_size))
        )
        shift_labels = eqx.error_if(
            shift_labels,
            has_invalid_labels,
            f"labels contain out-of-bounds values. Valid range: [0, {vocab_size})",
        )

        safe_labels = jnp.where(valid_mask, shift_labels, 0)

        shift_logits_softmax = shift_logits.astype(softmax_dtype)
        log_probs = jax.nn.log_softmax(shift_logits_softmax, axis=-1)
        batch_idx = jnp.arange(log_probs.shape[0])[:, None]
        seq_idx = jnp.arange(log_probs.shape[1])[None, :]
        target_log_probs = log_probs[batch_idx, seq_idx, safe_labels]

        target_log_probs = jnp.where(
            valid_mask, target_log_probs, jnp.zeros((), dtype=softmax_dtype)
        )
        num_valid = valid_mask.sum().astype(softmax_dtype)
        num_valid = jnp.maximum(num_valid, jnp.array(1.0, dtype=softmax_dtype))
        loss = -target_log_probs.sum() / num_valid
        return loss

    attn_mod.attention_single_chunk = attention_single_chunk
    attn_mod.attention_multi_chunk = attention_multi_chunk
    attn_mod.ChunkedAttention.__call__ = chunked_call
    attn_mod.MegalodonAttention.__call__ = attn_call
    model_mod._checkpointed_layer = _checkpointed_layer
    model_mod.MegalodonBlock.__call__ = block_call
    model_mod.MegalodonModel.__call__ = model_call
    model_mod.MegalodonForCausalLM.__call__ = lm_call
    model_mod.MegalodonForCausalLM.compute_loss = compute_loss

    attn_mod._CHOMP_SEGMENT_IDS_PATCHED = True
    return True
