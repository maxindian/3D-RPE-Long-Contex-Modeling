# -*- coding:utf-8 -*-

from typing import List, Optional, Tuple
from torch import nn
import math
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv
import torch
import transformers
from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_func
from transformers.cache_utils import Cache

def merge_attn_outputs(flash_results):
    attn_outputs_all = [flash_results[0][0]]
    flash_results = flash_results[1:]
    for flash_per_chunk in flash_results:
        attn_outputs = torch.stack([flash_attn_output[0] for flash_attn_output in flash_per_chunk])
        logits = torch.stack([flash_attn_output[1] for flash_attn_output in flash_per_chunk])
        max_logits = torch.max(logits, dim=0).values  
        stable_logits = logits - max_logits.unsqueeze(0)  

        lse_s = torch.exp(stable_logits).detach()
        lse_sum = torch.sum(lse_s, dim=0)
        lse_s /= lse_sum
        attn_outputs *= lse_s.unsqueeze(-1)
        attn_outputs_all.append(attn_outputs.sum(dim=0))
    return torch.cat(attn_outputs_all, dim=2)


def do_flash_attn(query_states, key_states, value_states, causal=True):
    # flash_attention
    output, softmax_lse, _ = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2),
                                             value_states.transpose(1, 2), causal=causal, return_attn_probs=True)
    return output.transpose(1, 2), softmax_lse


def apply_rotary_pos_emb(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_emb = (x * cos) + (rotate_half(x) * sin)
    return x_emb

def apply_bsrotary_pos_emb(x, d3_cos, d3_sin, chunk_len):
    x_seq_length = x.shape[-2]
    sum_chunk =  math.ceil(x_seq_length / chunk_len)
    for chunk_index in range(2, sum_chunk//2):
        rotary_pos_emb = x[:, :, chunk_len*(chunk_index):chunk_len*(chunk_index+2), :]
        rotary_pos_emb = (rotary_pos_emb * d3_cos[chunk_index-1]) + (rotate_half(rotary_pos_emb) * d3_sin[chunk_index-1])
        x[:, :, chunk_len*chunk_index:chunk_len*(chunk_index+2), :] = rotary_pos_emb
    return x

def apply_bsrotary_q_emb(x, d3_cos, d3_sin, scale_number):
    rotary_q_emb = (x * d3_cos[scale_number]) + (rotate_half(x) * d3_sin[scale_number])
    return rotary_q_emb

class CPERotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, scaling_factor=1.0, device=None):
        super().__init__() 
        self.max_seq_len = max_position_embeddings
        self.space_rotary_dim = 100
        self.dim = dim
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = max_position_embeddings
        self.base = base
    
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.max_seq_len,
            device=device, dtype=torch.float32
        )
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        inv_freq_bs = 1.0 / (self.base ** (torch.arange(0, self.space_rotary_dim).float().to(device)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("inv_freq_bs", inv_freq_bs, persistent=False)

        chunk_len = chunk_size - local_window
        q_t = torch.arange(chunk_len, device=device, dtype=self.inv_freq.dtype) / self.scaling_factor
        qc_t = (torch.arange(chunk_len, device=device, dtype=self.inv_freq.dtype) + chunk_len).clamp(max=chunk_size) / self.scaling_factor
        k_t = (torch.arange(seq_len + MAX_NEW_TOKENS, device=device, dtype=self.inv_freq.dtype) % chunk_len) / self.scaling_factor
        
        q_freqs = torch.outer(q_t, self.inv_freq)   # seq_len x dim/2
        qc_freqs = torch.outer(qc_t, self.inv_freq) # seq_len x dim/2
        k_freqs = torch.outer(k_t, self.inv_freq)   # seq_len x dim/2
        

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        q_emb = torch.cat((q_freqs, q_freqs), dim=-1)  # seq_len x dim
        qc_emb = torch.cat((qc_freqs, qc_freqs), dim=-1) # seq_len x dim
        k_emb = torch.cat((k_freqs, k_freqs), dim=-1)  # seq_len x dim
        
        self.register_buffer("qk3d_cos_cached", self.inv_freq_bs.cos().to(dtype), persistent=False)
        self.register_buffer("qk3d_sin_cached", self.inv_freq_bs.sin().to(dtype), persistent=False)
        self.register_buffer("q_cos_cached", q_emb.cos().to(dtype), persistent=False)
        self.register_buffer("q_sin_cached", q_emb.sin().to(dtype) , persistent=False)
        self.register_buffer("qc_cos_cached", qc_emb.cos().to(dtype) , persistent=False)
        self.register_buffer("qc_sin_cached", qc_emb.sin().to(dtype) , persistent=False)
        self.register_buffer("k_cos_cached", k_emb.cos().to(dtype), persistent=False)
        self.register_buffer("k_sin_cached", k_emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len=seq_len, device=self.inv_freq.device, dtype=torch.get_default_dtype())
            self.max_seq_len = seq_len
        return (
            self.qk3d_cos_cached[:self.space_rotary_dim].to(dtype=x.dtype),
            self.qk3d_sin_cached[:self.space_rotary_dim].to(dtype=x.dtype),
            self.q_cos_cached[:seq_len].to(dtype=x.dtype),
            self.q_sin_cached[:seq_len].to(dtype=x.dtype),
            self.qc_cos_cached[:seq_len].to(dtype=x.dtype),
            self.qc_sin_cached[:seq_len].to(dtype=x.dtype),
            self.k_cos_cached[:seq_len].to(dtype=x.dtype),
            self.k_sin_cached[:seq_len].to(dtype=x.dtype),
        )

def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    chunk_len = chunk_size - local_window

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    # during inference
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # kv_seq_len += past_key_value[0].get_usable_length(kv_seq_len, self.layer_idx)
    
    q_seq_len = query_states.shape[-2]
    has_kv_cache = q_seq_len != kv_seq_len
    
    d3_cos, d3_sin, q_cos, q_sin, qc_cos, qc_sin, k_cos, k_sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    key_states = apply_rotary_pos_emb(key_states, k_cos, k_sin, position_ids)
    key_states = apply_bsrotary_pos_emb(key_states, d3_cos, d3_sin, chunk_len) # fine-tuning
    position_ids = position_ids % chunk_len   # slide
    if past_key_value is not None:
        # reuse k, v, self_attention
        # key_states = torch.cat([past_key_value[0], key_states], dim=2)
        # value_states = torch.cat([past_key_value[1], value_states], dim=2)
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs=None)
    
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    flash_results = []
    if not has_kv_cache:
        q_states_prev = apply_rotary_pos_emb(query_states[:, :, :chunk_len, :], q_cos, q_sin,
                                            position_ids[:, :chunk_len])
        k_states_prev = key_states[:, :, :chunk_len, :]
        v_states_prev = value_states[:, :, :chunk_len, :]
        flash_results.append(do_flash_attn(q_states_prev, k_states_prev, v_states_prev))
        remain_len = kv_seq_len - chunk_len 
        while remain_len > 0:
            flash_per_chunk = []
            begin = kv_seq_len - remain_len   # start position 
            curr_chunk_len = min(chunk_len, remain_len)
            end = begin + curr_chunk_len
            q_states_inner = apply_rotary_pos_emb(query_states[:, :, begin:end, :], q_cos, q_sin,
                                                position_ids[:, begin:end])
            # if begin > chunk_len:
            #     scale_number = math.floor(begin / chunk_len)
            #     k_states_inner = apply_bsrotary_q_emb(q_states_intra, d3_cos, d3_sin, scale_number)
            k_states_inner = key_states[:, :, begin:end, :]
            v_states_inner = value_states[:, :, begin:end, :]
            flash_per_chunk.append(do_flash_attn(q_states_inner, k_states_inner, v_states_inner))

            q_states_succ = apply_rotary_pos_emb(query_states[:, :, begin:end, :], qc_cos, qc_sin,
                                                  position_ids[:, begin:end])
            # if begin > chunk_len:
            #     scale_number = math.floor(begin / chunk_len)
            #     q_states_succ = apply_bsrotary_q_emb(q_states_succ, d3_cos, d3_sin, scale_number)

            flash_per_chunk.append(do_flash_attn(q_states_succ, k_states_prev, v_states_prev, False))

            if begin - (k_states_prev.size(-2)) > 0:
                prev_len = k_states_prev.size(-2)
                q_states_inter = apply_rotary_pos_emb(query_states[:, :, begin:end, :], qc_cos, qc_sin,
                                                    position_ids[:, chunk_len - 1][:, None].repeat(1, curr_chunk_len))
                scale_number = math.floor(begin / chunk_len)
                q_states_inter = apply_bsrotary_q_emb(q_states_inter, d3_cos, d3_sin, scale_number-1)
                k_states_inter = key_states[:, :, :begin - prev_len, :]
                v_states_inter = value_states[:, :, :begin - prev_len, :]
                flash_per_chunk.append(do_flash_attn(q_states_inter, k_states_inter, v_states_inter, False))
        
            flash_results.append(flash_per_chunk)
            k_states_prev = k_states_inner
            v_states_prev = v_states_inner
            remain_len = remain_len - chunk_len

        attn_output = merge_attn_outputs(flash_results)
    else:
        chunk_num_curr = (kv_seq_len - 1) // chunk_len
        q_states_intra = apply_rotary_pos_emb(query_states, q_cos, q_sin, position_ids)
        k_states_intra = key_states[:, :, chunk_len * chunk_num_curr:kv_seq_len, :]
        attn_weights = torch.matmul(q_states_intra, k_states_intra.transpose(2, 3)) / math.sqrt(
            self.head_dim)
        attn_scores = [attn_weights]
        if chunk_num_curr >=1:
            q_states_succ = apply_rotary_pos_emb(query_states, qc_cos, qc_sin, position_ids)

            k_states_succ = key_states[:, :, chunk_len * (chunk_num_curr - 1):chunk_len * chunk_num_curr, :]
            attn_weights = torch.matmul(q_states_succ, k_states_succ.transpose(2, 3)) / math.sqrt(
                self.head_dim)
            attn_scores = [attn_weights] + attn_scores

        if chunk_num_curr >= 2:
            q_states_inter = apply_rotary_pos_emb(query_states, qc_cos, qc_sin,
                                                torch.tensor([[chunk_len - 1]], device=query_states.device))
            q_states_inter = apply_bsrotary_q_emb(q_states_inter, d3_cos, d3_sin, chunk_num_curr-1)
            k_states_inter = key_states[:, :, :chunk_len * (chunk_num_curr - 1), :]
            attn_weights = torch.matmul(q_states_inter, k_states_inter.transpose(2, 3)) / math.sqrt(
                self.head_dim)
            attn_scores = [attn_weights] + attn_scores 

        attn_weights = torch.cat(attn_scores, dim=-1)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

chunk_size = None
local_window = None
linear_factor = None
MAX_NEW_TOKENS = 512

def replace_with_cpellama(pretraining_length=4096, local_window_size=None):
    global chunk_size
    global local_window
    chunk_size = pretraining_length * 3 // 4
    local_window = local_window_size if local_window_size else pretraining_length // 16
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = forward
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = CPERotaryEmbedding
    transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding = CPERotaryEmbedding