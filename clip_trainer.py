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

def pmap_batch(batch):
    """Splits the first axis of `arr` evenly across the number of devices."""
    per_device_batch_size = batch.shape[0] // jax.device_count()
    batch = batch[:per_device_batch_size * jax.device_count()] # trim the rest of the batch
    return batch.reshape(jax.device_count(), per_device_batch_size, *batch.shape[1:])

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
        
        @partial(jax.pmap, axis_name='dp')
        def eval_pmap(params, x, y):
            loss = train_loss(params, x, y)
            return jax.lax.pmean(loss, axis_name='dp')

        @partial(jax.pmap, axis_name='dp')
        def train_pmap(params, opt_state, x, y):
            val_grad_fn = jax.value_and_grad(train_loss)
            loss, grad = val_grad_fn(params, x, y)

            grads = jax.lax.pmean(grad, axis_name='dp')
            loss = jax.lax.pmean(loss, axis_name='dp')

            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return loss, params, opt_state
                
        self.train_pmap = train_pmap
        self.eval_pmap = eval_pmap
        self.state = init(next(key), x)
        self.eval_weights = None

        param_count = hk.data_structures.tree_size(self.state['params'])
        print(f"Total parameters: {param_count}")

    def train(self, sample):
        obs, target = map(pmap_batch, (sample["obs"], sample["target"]))
        loss, params, opt_state = self.train_pmap(self.state["params"], self.state["opt_state"], obs, target)
        self.state = {
                "params": params,
                "step": self.state["step"] + 1,
                "opt_state": opt_state,
            }
        return np.array(loss).mean()

    def eval(self, sample):
        obs, target = map(pmap_batch, (sample["obs"], sample["target"]))
        loss = self.eval_pmap(self.state["params"], obs, target)
        return np.array(loss).mean()
