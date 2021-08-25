import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import optax

from jax.experimental.maps import mesh
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit

from clip_model import TextCLIP
import clip_jax 

def cfg_encode_text(config, tokens):
    clip = TextCLIP( # the same as orig, except context_length = 77 - 2 (special tokens)
                     embed_dim = 512, 
                     context_length = 75, 
                     vocab_size = 49408,
                     # can possibly vary
                     rotary_dims = config["rotary_dims"], 
                     transformer_width = config["d_model"],
                     transformer_heads = config["n_heads"],
                     transformer_layers = config["layers"])
    return clip.encode_text(tokens)

class ClipTrainer:
    def __init__(self, config):
        self.config = config
        optimizer = config["optimizer"]
        
        _, clip_target, _, _ = clip_jax.load('ViT-B/32', "cpu")
        
        encode_text = partial(cfg_encode_text,config)
        clip_init_fn = hk.transform(hk.experimental.optimize_rng_use(encode_text)).init
            
        def init(key, xs):
            params = clip_init_fn(key, tokens = xs)
            opt_state = optimizer.init(params)
            
            return {
                "params": params,
                "step": np.array(0),
                "opt_state": opt_state
            }

        key = hk.PRNGSequence(42)
        x = jax.random.randint(next(key), (jax.local_device_count(),75), 0, 49408)
        
        clip_apply_fn = hk.without_apply_rng(hk.transform(encode_text)).apply

        def train_loss(params, x, y):
            return jnp.mean(jnp.square(clip_apply_fn(params, x) - clip_target(y)))
            
        def train(state, x, y):
            val_grad_fn = jax.value_and_grad(train_loss)
            loss, grad = val_grad_fn(state["params"], x, y)
            updates, new_opt_state = optimizer.update(grad, state["opt_state"], state["params"])
            
            return loss, {
                "params": optax.apply_updates(state["params"], updates),
                "step": state["step"] + 1,
                "opt_state": new_opt_state,
            }
        
        PS=PartitionSpec('devices')
        self.train_pjit = pjit(train,
                               in_axis_resources=(None, PS, PS),
                               out_axis_resources=(None))
        #self.train_pjit = train
        self.eval_pjit = pjit(train_loss,
                              in_axis_resources=(None, PS, PS),
                              out_axis_resources=(None))
        #self.eval_pjit = train_loss
        self.state = init(next(key), x)
        self.eval_weights = None

        param_count = hk.data_structures.tree_size(self.state['params'])
        print(f"Total parameters: {param_count}")

    def train(self, sample):
        obs = sample["obs"]
        target = sample["target"]
        #print(f"shapes {obs.shape} {target.shape}" )
        loss, self.state = self.train_pjit(self.state, obs, target)
        loss = np.array(loss)

        return loss.mean()

    def eval(self, sample):
        out = self.eval_pjit(self.state["params"], sample["obs"], sample["target"])
        return out