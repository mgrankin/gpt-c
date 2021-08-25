import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from haiku import LayerNorm
from einops import rearrange, repeat
from functools import lru_cache, partial

# there is no limit for lenght of text, because our countext windows is always 75 tokens, 
# but we have to choose for RoPE
MAX_SEQ_LEN = 8192

@lru_cache()
def fixed_pos_embedding(rotary_dims):
    inv_freq = 1. / (10000 ** (np.arange(0, rotary_dims, 2) / rotary_dims))
    sinusoid_inp = np.einsum('i , j -> i j', np.arange(MAX_SEQ_LEN), inv_freq)
    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(x, sincos, seq_dim):
    sincos = map(lambda t: repeat(t, '... b n -> ... b (n j)', j=2)[-x.shape[seq_dim]:], sincos)
    
    # (n_seq, dim_per_head) -> (n_seq, 1, 1, dim_per_head), so we can do mult
    # in case "x" is something like (n_seq, bs, n_heads, dim_per_head)
    add_dims = set(np.arange(x.ndim-1)) - set([np.arange(x.ndim)[seq_dim]])
    sin, cos = map(lambda t: jnp.expand_dims(t, tuple(add_dims)), sincos)
    
    return (x * cos) + (rotate_every_two(x) * sin)

@partial(jax.jit, static_argnums=(1,2))
def apply_rope(x, rotary_dims, seq_dim):
    x_rot = x[..., :rotary_dims]
    x_pass = x[..., rotary_dims:]
    sincos = fixed_pos_embedding(rotary_dims)
    x_rot = apply_rotary_pos_emb(x_rot, sincos, seq_dim)
    return jnp.concatenate([x_rot, x_pass], axis=-1)

def rope_tests():
    rotary_dims = 32
    vectors = np.random.random(size=(2,75))
    def test_pos(pos1, pos2, f):
        q = np.zeros(shape=(1,64,75))
        v = np.zeros(shape=(1,64,75))
        q[0,pos1] = vectors[0]
        v[0,pos2] = vectors[1]
        res = f(q,rotary_dims)@f(v,rotary_dims).transpose(0, 2, 1)
        return res[0,pos1,pos2]
    
    pos0 = test_pos(3,17, lambda x,y: x)
    pos1 = test_pos(3,17, apply_rope)
    pos2 = test_pos(5,19, apply_rope)
    pos3 = test_pos(5,20, apply_rope)
    assert not jnp.isclose(pos0, pos1)
    assert jnp.isclose(pos1, pos2)
    assert not jnp.isclose(pos2, pos3)

#rope_tests()

def rope_tests2():
    rotary_dims = 32
    vectors = np.random.random(size=(2,75))
    def test_pos(pos1, pos2, f):
        q = np.zeros(shape=(64,75))
        v = np.zeros(shape=(64,75))
        q[pos1] = vectors[0]
        v[pos2] = vectors[1]
        q = q[:,None,None,:]
        v = v[:,None,None,:]
        q = f(q,rotary_dims,0).transpose(1,2,0,3)
        v = f(v,rotary_dims,0).transpose(1,2,0,3)
        q = jnp.squeeze(q)
        q = jnp.squeeze(q)
        v = jnp.squeeze(v)
        v = jnp.squeeze(v)
        res = q@v.transpose()
        return res[pos1,pos2]
    
    pos0 = test_pos(3,17, lambda x,y,z: x)
    pos1 = test_pos(3,17, apply_rope)
    pos2 = test_pos(5,19, apply_rope)
    pos3 = test_pos(5,20, apply_rope)
    assert not jnp.isclose(pos0, pos1)
    assert jnp.isclose(pos1, pos2)
    assert not jnp.isclose(pos2, pos3)

#rope_tests2()

class MultiHeadAttention(hk.Module):
    def __init__(
            self,
            num_heads: int,
            head_size: int,
            rotary_dims: int, 
            w_init_scale: float,
            attn_mask: jnp.ndarray = None,
            name: str = "mha",
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.model_size = head_size * num_heads
        self.w_init = hk.initializers.VarianceScaling(w_init_scale)
        self.attn_mask = attn_mask
        self.rotary_dims = rotary_dims

        self.in_proj_weight = hk.get_parameter("in_proj_weight", shape=[self.model_size * 3, self.model_size], init=self.w_init)
        self.in_proj_bias = hk.get_parameter("in_proj_bias", shape=[self.model_size * 3], init=self.w_init)
        self.out_proj = hk.Linear(self.model_size, name="out_proj")

    def __call__(
            self,
            x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute (optionally masked) MHA with queries, keys & values."""
        all_out = jnp.dot(x, self.in_proj_weight.transpose())
        all_out += self.in_proj_bias

        q, k, v = jnp.array_split(all_out, 3, axis=-1)

        query_heads = self._split(q)
        key_heads = self._split(k)
        value_heads = self._split(v)
        # RoPE
        query_heads = apply_rope(query_heads, self.rotary_dims, 0)
        key_heads = apply_rope(key_heads, self.rotary_dims, 0)
        
        attention_logits = jnp.einsum("tbhd,Tbhd->bhtT", query_heads, key_heads)
        sqrt_key_size = np.sqrt(self.model_size//self.num_heads).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size

        if self.attn_mask is not None:
            attention_logits += self.attn_mask

        attention_weights = jax.nn.softmax(attention_logits)
        attention = jnp.einsum("bhtT,Tbhd->tbhd", attention_weights, value_heads)
        # Concatenate attention matrix of all heads into a single vector.
        attention_vec = jnp.reshape(attention, (*q.shape[:2], -1))

        return self.out_proj(attention_vec)

    def _split(
            self,
            x: jnp.ndarray,
    ) -> jnp.ndarray:
        return x.reshape((*x.shape[:2], self.num_heads, self.model_size//self.num_heads))


class QuickGELU(hk.Module):
    def __call__(self, x: jnp.ndarray):
        return x * jax.nn.sigmoid(1.702 * x)


class ResidualAttentionBlock(hk.Module):
    def __init__(self, d_model: int, n_head: int, rotary_dims:int, attn_mask: jnp.ndarray, name: str):
        super().__init__(name=name)
        self.attn = MultiHeadAttention(n_head, d_model // n_head, rotary_dims, 1, attn_mask, name="attn")
        self.ln_1 = LayerNorm(-1, create_scale=True, create_offset=True, name="ln_1")
        with hk.experimental.name_scope("mlp"):
            self.mlp = [hk.Linear(d_model * 4, name="c_fc"),
                        QuickGELU(),
                        hk.Linear(d_model, name="c_proj")]

        self.ln_2 = LayerNorm(-1, create_scale=True, create_offset=True, name="ln_2")

    def run_mlp(self, x: jnp.ndarray):
        for f in self.mlp:
            x = f(x)
        return x

    def __call__(self, x: jnp.ndarray):
        x = x + self.attn(self.ln_1(x))
        x = x + self.run_mlp(self.ln_2(x))
        return x


class Transformer(hk.Module):
    def __init__(self, width: int, layers: int, heads: int, rotary_dims: int, name: str, attn_mask=None):
        super().__init__(name=name)
        self.width = width
        self.layers = layers
        self.resblocks = [ResidualAttentionBlock(width, heads, rotary_dims, attn_mask, name=f"resblocks{i}") for i in range(layers)]
        self.attn_mask = attn_mask

    def __call__(self, x: jnp.ndarray):
        for b in self.resblocks:
            x = b(x)
        return x

class TextCLIP(hk.Module):
    @hk.transparent
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 rotary_dims: int, 
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 seq_length:int = None
                 ):
        super().__init__()

        self.context_length = context_length
        if seq_length is None:
            seq_length = context_length
        self.seq_length = seq_length
            
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            rotary_dims=rotary_dims,
            attn_mask=self.build_attention_mask(),
            name="transformer"
        )

        self.vocab_size = vocab_size
        self.token_embedding = hk.Embed(vocab_size, transformer_width, name="token_embedding")

        scale = transformer_width ** -0.5
        w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(scale))
        self.ln_final = LayerNorm(-1, create_scale=True, create_offset=True, name="ln_final")

        self.text_projection = hk.get_parameter("text_projection", shape=[transformer_width, embed_dim], init=w_init)
        self.logit_scale = hk.get_parameter("logit_scale", shape=[], init=hk.initializers.Constant(1))

    def build_attention_mask(self):
        # we use additive attention mask; fill with -inf
        mask = jnp.zeros((self.seq_length, self.seq_length))
        mask -= 10e10
        # make zeroes in place of context windows, -inf otherwise
        mask = jnp.triu(mask, self.context_length).transpose() + jnp.triu(mask, 1)
        return mask

    def encode(self, text):
        x = self.token_embedding(text)  # [batch_size, d_input, d_model]

        x = x.transpose((1, 0, 2))  # NLD -> LND

        x = self.transformer(x)
        x = x.transpose((1, 0, 2))  # LND -> NLD
        x = self.ln_final(x) @ self.text_projection
        return x
    
    def encode_text(self, text):
        x = self.encode(text)
        # x.shape == [batch_size, n_ctx, transformer.width]
        # take features from the last non-zero token 
        pos = jnp.cumsum(text, axis=-1).argmax(axis=-1)
        x = x[jnp.arange(x.shape[0]), pos] 
        return x

