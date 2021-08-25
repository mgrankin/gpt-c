import hashlib
import os
import urllib
import warnings
from typing import Union, List

import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image
from haiku._src.data_structures import FlatMapping
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import CLIP, get_params
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from functools import partial

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    # only ViT is supported for now
    # "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        lambda tensor: tensor.cpu().detach().numpy()
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def convert_params(torch_state, jax_params):
    def name_iter(pytree, root, f):
        new_out = {}
        for k, v in pytree.items():
            if isinstance(v, FlatMapping):
                new_out[k] = name_iter(v, root + "/" + k, f)
            else:
                new_out[k] = f(v, root + "/" + k)
        return new_out

    def process_node(value, name):
        name = name.lstrip("/")
        tensor_name = name.split("/")[-1]
        tensor_name = {
            "w": "weight",
            "b": "bias",
            "scale": "weight",
            "offset": "bias",
            "embeddings": "weight",
        }.get(tensor_name, tensor_name)

        tensor_path = "/".join(name.split("/")[:-1]).replace("/~/", ".").replace("/", ".").replace("resblocks",
                                                                                                   "resblocks.").replace(
            "~", "")
        new_tensor = value

        pytorch_name = tensor_path + "." + tensor_name if tensor_path else tensor_name

        if "conv" in name:
            pytorch_path = tensor_path + "." + tensor_name
            pytorch_tensor = torch_state[pytorch_path].permute([2, 3, 1, 0])
            new_tensor = jnp.array(pytorch_tensor)
        elif pytorch_name in torch_state:
            pytorch_tensor = torch_state[pytorch_name]

            if tensor_name == "weight" and "token_embedding" not in tensor_path:
                pytorch_tensor = pytorch_tensor.t()

            new_tensor = jnp.array(pytorch_tensor)
        else:
            raise Exception("not implemented")

        assert new_tensor.shape == value.shape
        return new_tensor.astype("float32")

    return name_iter(jax_params, "", process_node)


def load(name: str, device: Union[str, torch.device] = "cpu", jit=True):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        state_dict = torch.jit.load(model_path, map_location=device if jit else "cpu").eval().state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    clip_params = get_params(state_dict)

    # jax model
    def clip_jax(image, text):
        clip = CLIP(**clip_params)
        return clip.encode_image(image), clip.encode_text(text)

    def vit_jax(image):
        clip = CLIP(**clip_params)
        return clip.encode_image(image)

    def text_jax(text):
        clip = CLIP(**clip_params)
        return clip.encode_text(text)
    
    rng_key = jax.random.PRNGKey(42)
    transformed = hk.transform(clip_jax)
    jax_params = transformed.init(rng=rng_key, image=jnp.zeros((1, 3, 224, 224)), text=jnp.zeros((1, 77), dtype=jnp.int16))
    jax_params = convert_params(state_dict, jax_params)

    image_fn = hk.without_apply_rng(hk.transform(vit_jax)).apply
    text_fn = hk.without_apply_rng(hk.transform(text_jax)).apply

    return partial(image_fn, jax_params), partial(text_fn, jax_params), jax_params, _transform(clip_params["image_resolution"])


def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), context_length), dtype=np.int32)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = tokens

    return result
