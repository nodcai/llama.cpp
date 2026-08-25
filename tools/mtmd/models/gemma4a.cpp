/**
 * Gemma 4 Audio Conformer Encoder (clip_graph_gemma4a)
 *
 * Architecture: Conformer with dual half-step FFN, chunked-local self-attention
 * with sinusoidal RPE and softcap, depthwise causal conv module, output
 * projection, and a soft-token mm embedder.
 *
 * The graph is split into three reusable pieces — stem / body / head — so the
 * streaming variant can share stem and head with the one-shot path while
 * swapping the body for a chunk-at-a-time version with explicit past state.
 */

#include "models.h"
#include <cmath>

ggml_tensor * clip_graph_gemma4a::build_stem(ggml_tensor * mel_in) {
    const float norm_eps = 1e-6f;

    auto * cur = ggml_cont(ctx0, ggml_transpose(ctx0, mel_in));

    // Subsampling Conv2D (symmetric padding=1, matching PyTorch)
    for (int i = 0; i < 2; i++) {
        cur = ggml_conv_2d(ctx0, model.sscp_conv_w[i], cur, 2, 2, 1, 1, 1, 1);
        if (model.sscp_conv_b[i]) {
            cur = ggml_add(ctx0, cur, model.sscp_conv_b[i]);
        }
        // nn.LayerNorm(channels): permute ch to ne[0], normalize, permute back
        if (model.sscp_norm_w[i]) {
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
            cur = ggml_norm(ctx0, cur, norm_eps);
            cur = ggml_mul(ctx0, cur, model.sscp_norm_w[i]);
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));
        }
        cur = ggml_relu(ctx0, cur);
    }
    // Flatten [freq, time, ch, 1] -> [ch*freq, time]
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);
    if (model.sscp_inp_proj_w) {
        cur = build_mm(model.sscp_inp_proj_w, cur);
        if (model.sscp_inp_proj_b) {
            cur = ggml_add(ctx0, cur, model.sscp_inp_proj_b);
        }
    }
    return cur;
}

ggml_tensor * clip_graph_gemma4a::build_body(ggml_tensor * input) {
    const float res_weight = 0.5f;
    const float norm_eps   = 1e-6f;

    auto * cur = input;

    const int64_t n_pos = cur->ne[1];

    // Chunked local attention parameters
    const int64_t C  = 12;                              // chunk_size
    const int64_t P  = 12;                              // max_past_horizon (context_left - 1)
    const int64_t S  = C + P;                           // context_size = 24
    const int64_t R  = P + 1;                           // RPE positions = 13
    const int64_t B  = (n_pos + C - 1) / C;            // num_blocks
    const int64_t Np = B * C;                           // padded sequence length
    const int64_t pad_seq = Np - n_pos;

    // Input tensors: blocked RPE and blocked attention mask
    ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_head * d_head, R);
    ggml_set_name(pos_emb, "pos_emb");
    ggml_set_input(pos_emb);

    ggml_tensor * kq_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, S, C, B);
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_input(kq_mask);

    // Conformer Blocks
    for (int il = 0; il < hparams.n_layer; il++) {
        const auto & layer = model.layers[il];
        auto * residual = cur;

        // FFN 1 (half-step)
        if (layer.ff_norm_w && layer.ff_up_w && layer.ff_down_w) {
            cur = build_norm(cur, layer.ff_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            cur = build_ffn(cur,
                layer.ff_up_w, nullptr, nullptr, nullptr,
                layer.ff_down_w, nullptr, FFN_SILU, il);
            if (layer.ff_post_norm_w) {
                cur = build_norm(cur, layer.ff_post_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, res_weight));
        }

        // Chunked local self-attention with RPE
        if (layer.q_w && layer.k_w && layer.v_w && layer.o_w) {
            const float q_scale = (1.0f / sqrtf((float)d_head)) / logf(2.0f);
            const float k_scale = logf(1.0f + expf(1.0f)) / logf(2.0f);
            const float softcap = 50.0f;

            ggml_tensor * attn_norm_w = layer.attn_pre_norm_w ? layer.attn_pre_norm_w : layer.ln_1_w;
            cur = attn_norm_w
                ? build_norm(residual, attn_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il)
                : residual;

            ggml_tensor * Qcur = build_mm(layer.q_w, cur);
            ggml_tensor * Kcur = build_mm(layer.k_w, cur);
            ggml_tensor * Vcur = build_mm(layer.v_w, cur);

            // [n_embd, n_pos] -> [D, H, N]
            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

            // Q/K scaling
            Qcur = ggml_scale(ctx0, Qcur, q_scale);
            if (layer.per_dim_scale_w) {
                Qcur = ggml_mul(ctx0, Qcur, ggml_reshape_3d(ctx0, layer.per_dim_scale_w, d_head, 1, 1));
            }
            Kcur = ggml_scale(ctx0, Kcur, k_scale);
            if (layer.per_dim_k_scale_w) {
                Kcur = ggml_mul(ctx0, Kcur, ggml_reshape_3d(ctx0, layer.per_dim_k_scale_w, d_head, 1, 1));
            }

            // Q blocking: [D, H, N] -> pad to Np -> reshape [D, H, C, B]
            // ggml permute: ne[ax_i] = src->ne[i], so (0,3,1,2) sends H->3, C->1, B->2
            Qcur = ggml_pad(ctx0, Qcur, 0, 0, pad_seq, 0);          // [D, H, Np]
            Qcur = ggml_reshape_4d(ctx0, Qcur, d_head, n_head, C, B); // [D, H, C, B]
            Qcur = ggml_cont(ctx0, ggml_permute(ctx0, Qcur, 0, 3, 1, 2)); // [D, C, B, H]

            // K/V block context extraction via overlapping view:
            // Left pad by P and right pad to S*B elements,
            // then view with stride C in the block dimension (overlapping windows).
            auto extract_blocks = [&](ggml_tensor * t) -> ggml_tensor * {
                // [D, H, N] -> left pad by P, right pad to S*B
                const int64_t pad_kv = S * B - n_pos;
                t = ggml_pad_ext(ctx0, t, 0, 0, 0, 0, P, pad_kv - P, 0, 0); // [D, H, S*B]
                // Overlapping view: stride for B dim is C positions, not S
                // ne = [D, H, S, B], data_size = D*H*S*B*sizeof = source_nbytes (exact fit)
                // nb1=D*sizeof, nb2=D*H*sizeof, nb3=C*D*H*sizeof (overlap: C < S)
                t = ggml_view_4d(ctx0, t, d_head, n_head, S, B,
                    t->nb[1], t->nb[2], C * t->nb[2], 0);
                t = ggml_cont(ctx0, t);                       // materialize overlapping windows
                return t;
            };

            ggml_tensor * Kblk = extract_blocks(Kcur);
            // [D, H, S, B] -> [D, S, B, H] via permute(0,3,1,2)
            Kblk = ggml_cont(ctx0, ggml_permute(ctx0, Kblk, 0, 3, 1, 2));

            ggml_tensor * Vblk = extract_blocks(Vcur);
            // [D, H, S, B] -> [S, D, B, H] via permute(1,3,0,2)
            Vblk = ggml_cont(ctx0, ggml_permute(ctx0, Vblk, 1, 3, 0, 2));

            // Content attention: Q @ K^T
            // Kblk=[D,S,B,H], Qcur=[D,C,B,H] -> mul_mat contracts on D -> [S,C,B,H]
            ggml_tensor * matrix_ac = ggml_mul_mat(ctx0, Kblk, Qcur);

            // Relative position attention
            if (layer.attn_k_rel_w) {
                // RPE: [n_embd, R] -> project -> [D, H, R] -> [D, R, H]
                auto * p = ggml_mul_mat(ctx0, layer.attn_k_rel_w, pos_emb);
                p = ggml_reshape_3d(ctx0, p, d_head, n_head, R);
                p = ggml_cont(ctx0, ggml_permute(ctx0, p, 0, 2, 1, 3)); // [D, R, H]

                // Q_flat @ RPE^T: [D, C*B, H] @ [D, R, H] -> [R, C*B, H]
                auto * Q_flat = ggml_reshape_3d(ctx0, Qcur, d_head, C * B, n_head);
                auto * matrix_bd = ggml_mul_mat(ctx0, p, Q_flat);       // [R, C*B, H]
                matrix_bd = ggml_reshape_4d(ctx0, matrix_bd, R, C, B, n_head); // [R, C, B, H]

                // Blocked relative shift (appendix B of Transformer-XL)
                {
                    matrix_bd = ggml_pad(ctx0, matrix_bd, S + 1 - R, 0, 0, 0); // [S+1, C, B, H]
                    matrix_bd = ggml_reshape_3d(ctx0, matrix_bd, (S + 1) * C, B, n_head);
                    matrix_bd = ggml_view_3d(ctx0, matrix_bd,
                        C * S, B, n_head,
                        matrix_bd->nb[1], matrix_bd->nb[2], 0);
                    matrix_bd = ggml_cont(ctx0, matrix_bd);              // [C*S, B, H]
                    matrix_bd = ggml_reshape_4d(ctx0, matrix_bd, S, C, B, n_head); // [S, C, B, H]
                }

                matrix_ac = ggml_add(ctx0, matrix_ac, matrix_bd);
            }

            auto * scores = matrix_ac; // [S, C, B, H]

            // Softcap
            scores = ggml_scale(ctx0, scores, 1.0f / softcap);
            scores = ggml_tanh(ctx0, scores);
            scores = ggml_scale(ctx0, scores, softcap);

            // Blocked attention mask: [S, C, B] broadcasts over H
            scores = ggml_add(ctx0, scores, kq_mask);

            ggml_tensor * attn = ggml_soft_max(ctx0, scores);

            // attn @ V: [S,C,B,H] @ [S,D,B,H] -> [D,C,B,H]
            ggml_tensor * x = ggml_mul_mat(ctx0, Vblk, attn);

            // [D,C,B,H] -> [D,H,C,B] via permute(0,2,3,1) -> flatten -> trim
            x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 3, 1));
            x = ggml_cont_2d(ctx0, x, d_head * n_head, C * B);
            if (pad_seq > 0) {
                x = ggml_view_2d(ctx0, x, d_head * n_head, n_pos, x->nb[1], 0);
                x = ggml_cont(ctx0, x);
            }

            x = build_mm(layer.o_w, x);
            if (layer.o_b) { x = ggml_add(ctx0, x, layer.o_b); }

            if (layer.attn_post_norm_w) {
                x = build_norm(x, layer.attn_post_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, x);
        }

        // Convolution Module
        if (layer.norm_conv_w && layer.conv_pw1_w && layer.conv_dw_w && layer.conv_pw2_w) {
            cur = build_norm(residual, layer.norm_conv_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            auto * x = build_mm(layer.conv_pw1_w, cur);

            // GLU
            {
                int64_t d = x->ne[0] / 2;
                ggml_tensor * gate = ggml_sigmoid(ctx0,
                    ggml_cont(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], d * x->nb[0])));
                x = ggml_mul(ctx0,
                    ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], 0), gate);
                x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
            }

            // Causal depthwise Conv1D via ggml_ssm_conv, left padded only.
            x = ggml_pad_ext(ctx0, x, 4, 0, 0, 0, 0, 0, 0, 0);
            x = ggml_ssm_conv(ctx0, x, layer.conv_dw_w);
            if (layer.conv_dw_b) {
                x = ggml_add(ctx0, x, layer.conv_dw_b);
            }

            if (layer.conv_norm_w) {
                x = ggml_rms_norm(ctx0, x, norm_eps);
                x = ggml_mul(ctx0, x, layer.conv_norm_w);
            }
            x = ggml_silu(ctx0, x);
            x = build_mm(layer.conv_pw2_w, x);
            residual = ggml_add(ctx0, residual, x);
        }

        // FFN 2 (half-step)
        if (layer.ff_norm_1_w && layer.ff_up_1_w && layer.ff_down_1_w) {
            cur = build_norm(residual, layer.ff_norm_1_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            cur = build_ffn(cur,
                layer.ff_up_1_w, nullptr, nullptr, nullptr,
                layer.ff_down_1_w, nullptr, FFN_SILU, il);
            if (layer.ff_post_norm_1_w) {
                cur = build_norm(cur, layer.ff_post_norm_1_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, res_weight));
        }

        // Layer output norm
        cur = layer.ln_2_w
            ? build_norm(residual, layer.ln_2_w, nullptr, NORM_TYPE_RMS, norm_eps, il)
            : residual;
    }

    return cur;
}

ggml_tensor * clip_graph_gemma4a::build_head(ggml_tensor * encoded) {
    const float norm_eps = 1e-6f;
    auto * cur = encoded;

    // Output Projection
    if (model.audio_out_proj_w) {
        cur = build_mm(model.audio_out_proj_w, cur);
        if (model.audio_out_proj_b) {
            cur = ggml_add(ctx0, cur, model.audio_out_proj_b);
        }
    }

    // Audio Multimodal Embedder
    cur = ggml_rms_norm(ctx0, cur, norm_eps);
    if (model.mm_soft_emb_norm_w) {
        cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);
    }
    if (model.mm_input_proj_w) {
        cur = build_mm(model.mm_input_proj_w, cur);
    }
    return cur;
}

ggml_cgraph * clip_graph_gemma4a::build() {
    ggml_tensor * inp     = build_inp_raw(1);
    ggml_tensor * stem    = build_stem(inp);
    ggml_tensor * encoded = build_body(stem);
    ggml_tensor * head    = build_head(encoded);
    ggml_build_forward_expand(gf, head);
    return gf;
}

// ----- streaming variant -----
//
// Per-layer streaming attention with explicit past K/V (P=12 positions) and
// explicit past depthwise-conv state (4 positions of post-GLU activations).
// The chunked-attention math collapses to a single B=1 block: K/V window =
// concat(past_K_<il>, K_new) along position axis; mask + RPE shape are
// (S=24, C=12, B=1) so they share names/values with the one-shot input setter
// in clip.cpp. New state tensors are named so the caller can read them back
// after graph compute and re-feed them on the next chunk.

ggml_tensor * clip_graph_gemma4a::build_streaming_body(ggml_tensor * input) {
    const float res_weight = 0.5f;
    const float norm_eps   = 1e-6f;
    const float q_scale    = (1.0f / sqrtf((float)d_head)) / logf(2.0f);
    const float k_scale    = logf(1.0f + expf(1.0f)) / logf(2.0f);
    const float softcap    = 50.0f;

    auto * cur = input;

    // Streaming graph hard-codes the chunk geometry.
    const int64_t C = 12;       // chunk_size
    const int64_t P = 12;       // max_past
    const int64_t S = C + P;    // 24
    const int64_t R = P + 1;    // 13
    const int64_t B = 1;        // single block per call

    GGML_ASSERT(cur->ne[1] == C && "streaming body expects exactly 12 positions");

    stream_new_k.assign(hparams.n_layer, nullptr);
    stream_new_v.assign(hparams.n_layer, nullptr);
    stream_new_conv.assign(hparams.n_layer, nullptr);

    // Shared graph inputs (same names as the one-shot path so the existing
    // clip.cpp gemma4a setup code can populate them unchanged).
    ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_head * d_head, R);
    ggml_set_name(pos_emb, "pos_emb");
    ggml_set_input(pos_emb);

    ggml_tensor * kq_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, S, C, B);
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_input(kq_mask);

    char buf[64];

    for (int il = 0; il < hparams.n_layer; il++) {
        const auto & layer = model.layers[il];
        auto * residual = cur;

        // FFN 1 (half-step) — pointwise, no state.
        if (layer.ff_norm_w && layer.ff_up_w && layer.ff_down_w) {
            cur = build_norm(cur, layer.ff_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            cur = build_ffn(cur,
                layer.ff_up_w, nullptr, nullptr, nullptr,
                layer.ff_down_w, nullptr, FFN_SILU, il);
            if (layer.ff_post_norm_w) {
                cur = build_norm(cur, layer.ff_post_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, res_weight));
        }

        // Chunked attention with explicit past K/V.
        if (layer.q_w && layer.k_w && layer.v_w && layer.o_w) {
            ggml_tensor * attn_norm_w = layer.attn_pre_norm_w ? layer.attn_pre_norm_w : layer.ln_1_w;
            cur = attn_norm_w
                ? build_norm(residual, attn_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il)
                : residual;

            ggml_tensor * Qcur = build_mm(layer.q_w, cur);
            ggml_tensor * Kcur = build_mm(layer.k_w, cur);
            ggml_tensor * Vcur = build_mm(layer.v_w, cur);

            // [n_embd, C] -> [D, H, C]
            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, C);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, C);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, C);

            // Q scaling
            Qcur = ggml_scale(ctx0, Qcur, q_scale);
            if (layer.per_dim_scale_w) {
                Qcur = ggml_mul(ctx0, Qcur, ggml_reshape_3d(ctx0, layer.per_dim_scale_w, d_head, 1, 1));
            }
            // K scaling
            Kcur = ggml_scale(ctx0, Kcur, k_scale);
            if (layer.per_dim_k_scale_w) {
                Kcur = ggml_mul(ctx0, Kcur, ggml_reshape_3d(ctx0, layer.per_dim_k_scale_w, d_head, 1, 1));
            }

            // Past K/V graph inputs: [D, H, P]. Caller zero-initializes on
            // first chunk, replaces with new_k/v_<il> from the previous chunk
            // thereafter.
            snprintf(buf, sizeof buf, "past_k_%d", il);
            ggml_tensor * past_k = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d_head, n_head, P);
            ggml_set_name(past_k, buf);
            ggml_set_input(past_k);

            snprintf(buf, sizeof buf, "past_v_%d", il);
            ggml_tensor * past_v = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d_head, n_head, P);
            ggml_set_name(past_v, buf);
            ggml_set_input(past_v);

            // K/V windows: past then current along the position axis.
            ggml_tensor * Kwin = ggml_concat(ctx0, past_k, Kcur, 2);  // [D, H, S]
            ggml_tensor * Vwin = ggml_concat(ctx0, past_v, Vcur, 2);  // [D, H, S]

            // New state outputs (the K, V we just computed become past for
            // the next chunk). Use ggml_cpy into a fresh tensor so we have
            // a stable handle; expose via stream_new_k/_v for the caller's
            // forward-expand. The cpy must be reachable from a forward_expand
            // call to actually run.
            ggml_tensor * new_k_dst = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d_head, n_head, C);
            ggml_tensor * new_k_op  = ggml_cpy(ctx0, Kcur, new_k_dst);
            snprintf(buf, sizeof buf, "new_k_%d", il);
            ggml_set_name(new_k_op, buf);
            ggml_set_output(new_k_op);
            stream_new_k[il] = new_k_op;

            ggml_tensor * new_v_dst = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d_head, n_head, C);
            ggml_tensor * new_v_op  = ggml_cpy(ctx0, Vcur, new_v_dst);
            snprintf(buf, sizeof buf, "new_v_%d", il);
            ggml_set_name(new_v_op, buf);
            ggml_set_output(new_v_op);
            stream_new_v[il] = new_v_op;

            // Now build the standard chunked-attention math with B=1. The
            // existing one-shot code blocks K via the rolling/extract_blocks
            // trick; for B=1 we already have the right shape, just permute.
            //
            // Q: [D, H, C]      -> [D, C, B=1, H] (reshape_4d + permute)
            // Kwin: [D, H, S]   -> [D, S, B=1, H]
            // Vwin: [D, H, S]   -> [S, D, B=1, H]
            //
            // matrix_ac = mul_mat(Kwin, Q): contracts D, gives [S, C, B, H].
            ggml_tensor * Q4 = ggml_reshape_4d(ctx0, Qcur, d_head, n_head, C, B);
            Q4 = ggml_cont(ctx0, ggml_permute(ctx0, Q4, 0, 3, 1, 2)); // [D, C, B, H]

            ggml_tensor * K4 = ggml_reshape_4d(ctx0, Kwin, d_head, n_head, S, B);
            K4 = ggml_cont(ctx0, ggml_permute(ctx0, K4, 0, 3, 1, 2)); // [D, S, B, H]

            ggml_tensor * V4 = ggml_reshape_4d(ctx0, Vwin, d_head, n_head, S, B);
            V4 = ggml_cont(ctx0, ggml_permute(ctx0, V4, 1, 3, 0, 2)); // [S, D, B, H]

            ggml_tensor * matrix_ac = ggml_mul_mat(ctx0, K4, Q4);    // [S, C, B, H]

            // Relative position attention (same as build_body).
            if (layer.attn_k_rel_w) {
                auto * p = ggml_mul_mat(ctx0, layer.attn_k_rel_w, pos_emb);
                p = ggml_reshape_3d(ctx0, p, d_head, n_head, R);
                p = ggml_cont(ctx0, ggml_permute(ctx0, p, 0, 2, 1, 3)); // [D, R, H]

                auto * Q_flat = ggml_reshape_3d(ctx0, Q4, d_head, C * B, n_head);
                auto * matrix_bd = ggml_mul_mat(ctx0, p, Q_flat);                    // [R, C*B, H]
                matrix_bd = ggml_reshape_4d(ctx0, matrix_bd, R, C, B, n_head);       // [R, C, B, H]

                matrix_bd = ggml_pad(ctx0, matrix_bd, S + 1 - R, 0, 0, 0);            // [S+1, C, B, H]
                matrix_bd = ggml_reshape_3d(ctx0, matrix_bd, (S + 1) * C, B, n_head);
                matrix_bd = ggml_view_3d(ctx0, matrix_bd,
                    C * S, B, n_head,
                    matrix_bd->nb[1], matrix_bd->nb[2], 0);
                matrix_bd = ggml_cont(ctx0, matrix_bd);                                // [C*S, B, H]
                matrix_bd = ggml_reshape_4d(ctx0, matrix_bd, S, C, B, n_head);

                matrix_ac = ggml_add(ctx0, matrix_ac, matrix_bd);
            }

            auto * scores = matrix_ac;
            scores = ggml_scale(ctx0, scores, 1.0f / softcap);
            scores = ggml_tanh(ctx0, scores);
            scores = ggml_scale(ctx0, scores, softcap);
            scores = ggml_add(ctx0, scores, kq_mask);

            ggml_tensor * attn = ggml_soft_max(ctx0, scores);

            ggml_tensor * x = ggml_mul_mat(ctx0, V4, attn);          // [D, C, B, H]
            x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 3, 1)); // [D, H, C, B]
            x = ggml_cont_2d(ctx0, x, d_head * n_head, C * B);

            x = build_mm(layer.o_w, x);
            if (layer.o_b) { x = ggml_add(ctx0, x, layer.o_b); }

            if (layer.attn_post_norm_w) {
                x = build_norm(x, layer.attn_post_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, x);
        }

        // Convolution module with explicit past conv state. The one-shot
        // path pads the post-GLU tensor on the left with 4 zeros before the
        // depthwise conv; here we replace those zeros with the last 4
        // post-GLU positions from the previous chunk.
        if (layer.norm_conv_w && layer.conv_pw1_w && layer.conv_dw_w && layer.conv_pw2_w) {
            cur = build_norm(residual, layer.norm_conv_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            auto * x = build_mm(layer.conv_pw1_w, cur);

            // GLU (same as build_body): x ends as [T=C, conv_inner] with ne[0]=T.
            {
                int64_t d = x->ne[0] / 2;
                ggml_tensor * gate = ggml_sigmoid(ctx0,
                    ggml_cont(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], d * x->nb[0])));
                x = ggml_mul(ctx0,
                    ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], 0), gate);
                x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
            }
            // x layout: ne=[C, conv_inner], i.e., time-major.
            const int64_t conv_inner = x->ne[1];

            // Save the last 4 positions (along the time axis) as new conv
            // state. View shape [4, conv_inner] at time offset C-4 = 8.
            ggml_tensor * new_conv = ggml_cont(ctx0,
                ggml_view_2d(ctx0, x, 4, conv_inner, x->nb[1], (size_t)(C - 4) * x->nb[0]));
            snprintf(buf, sizeof buf, "new_conv_%d", il);
            ggml_set_name(new_conv, buf);
            ggml_set_output(new_conv);
            stream_new_conv[il] = new_conv;

            // Past conv state input: 4 positions of post-GLU activations from
            // the previous chunk. Same [4, conv_inner] layout as new_conv.
            snprintf(buf, sizeof buf, "past_conv_%d", il);
            ggml_tensor * past_conv = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 4, conv_inner);
            ggml_set_name(past_conv, buf);
            ggml_set_input(past_conv);

            // Replace the left zero-pad with the past state: cat along the
            // time axis (dim 0). ssm_conv (kernel 5) consumes 4 + C and
            // produces C.
            x = ggml_concat(ctx0, past_conv, x, 0);                  // [C + 4 = 16, conv_inner]
            x = ggml_ssm_conv(ctx0, x, layer.conv_dw_w);             // [C, conv_inner]
            if (layer.conv_dw_b) {
                x = ggml_add(ctx0, x, layer.conv_dw_b);
            }
            if (layer.conv_norm_w) {
                x = ggml_rms_norm(ctx0, x, norm_eps);
                x = ggml_mul(ctx0, x, layer.conv_norm_w);
            }
            x = ggml_silu(ctx0, x);
            x = build_mm(layer.conv_pw2_w, x);
            residual = ggml_add(ctx0, residual, x);
        }

        // FFN 2 (half-step)
        if (layer.ff_norm_1_w && layer.ff_up_1_w && layer.ff_down_1_w) {
            cur = build_norm(residual, layer.ff_norm_1_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            cur = build_ffn(cur,
                layer.ff_up_1_w, nullptr, nullptr, nullptr,
                layer.ff_down_1_w, nullptr, FFN_SILU, il);
            if (layer.ff_post_norm_1_w) {
                cur = build_norm(cur, layer.ff_post_norm_1_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, res_weight));
        }

        cur = layer.ln_2_w
            ? build_norm(residual, layer.ln_2_w, nullptr, NORM_TYPE_RMS, norm_eps, il)
            : residual;
    }

    return cur;
}

ggml_cgraph * clip_graph_gemma4a::build_streaming() {
    // The caller passes a mel slice covering only the encoder positions
    // through chunk K (the chunk to emit). stem yields positions [0, n_pos-1]
    // and the body operates on the last C=12 of those — i.e., chunk K.
    ggml_tensor * mel  = build_inp_raw(1);
    ggml_tensor * stem = build_stem(mel);

    const int64_t C = 12;
    GGML_ASSERT(stem->ne[1] >= C);

    ggml_tensor * chunk = ggml_view_2d(ctx0, stem,
        stem->ne[0], C, stem->nb[1],
        (size_t)(stem->ne[1] - C) * (size_t) stem->nb[1]);
    chunk = ggml_cont(ctx0, chunk);

    ggml_tensor * encoded = build_streaming_body(chunk);
    ggml_tensor * head    = build_head(encoded);
    // Name + mark the soft-token output explicitly. The caller must NOT rely on
    // it being the last graph node: the per-layer new_k/new_v/new_conv state
    // tensors are expanded after it below, so node(-1) is the last new_conv.
    ggml_set_name(head, "stream_out");
    ggml_set_output(head);
    ggml_build_forward_expand(gf, head);

    // The per-layer K/V/conv state-update tensors aren't on the head's
    // dependency path — expand them explicitly so they're computed.
    for (auto * t : stream_new_k)    if (t) ggml_build_forward_expand(gf, t);
    for (auto * t : stream_new_v)    if (t) ggml_build_forward_expand(gf, t);
    for (auto * t : stream_new_conv) if (t) ggml_build_forward_expand(gf, t);

    return gf;
}

ggml_tensor * clip_graph_gemma4a::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    auto it = model.clamp_info_map.find(w->name);
    if (it == model.clamp_info_map.end()) {
        return ggml_mul_mat(ctx0, w, x);
    }
    const auto & ci = it->second;
    ggml_tensor * clamped = ggml_clamp(ctx0, x, ci.inp_min, ci.inp_max);
    ggml_tensor * out = ggml_mul_mat(ctx0, w, clamped);
    return ggml_clamp(ctx0, out, ci.out_min, ci.out_max);
}
