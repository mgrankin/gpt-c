import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import wandb
from tqdm import tqdm

from clip_trainer import ClipTrainer
from tfrecord_loader import TFRecordNewInputs
from smart_open import open
from google.cloud import storage
from google.cloud.exceptions import NotFound
import pickle
import zstandard

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="""
    To use, download the full checkpoint archive, extract and upload to a GCS bucket, and set that as --tune-model-path
    Modify the config file:
        - set `model_dir` to where the checkpoints should be written during training
        - set `train_set`, `val_set` to index files for your data
        - set `warmup_steps`, `anneal_steps`, `lr`, `end_lr` to the lr schedule for your finetuning run
        - the global step will reset to 0, keep that in mind when writing your lr schedule
        - set `name` to specify the name of the Weights & Biases run
        - set `wandb_project` to specify the Weights & Biases project to log to
    To prepare data in the expected data format:
        - use the script `create_finetune_tfrecords.py` in this repo to create data in the expected format
        - upload the .tfrecords files to GCS
        - save their GCS paths to a index file under `data/`, see existing files for examples
    """,
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="Config file location")
    parser.add_argument("--tune-model-path", type=str, default=None, help="Base model to finetune")
    parser.add_argument("--fresh-opt", default=False, action="store_true", help="Use a newly initialized optimizer, ignoring any optimizer state saved in the base checkpoint")

    args = parser.parse_args()
    return args

def pmap_on(tree):
    return jax.tree_map(lambda x: jnp.array([x] * jax.device_count()), tree)

def pmap_off(tree):
    return jax.device_get(jax.tree_map(lambda x: x[0], tree))

def z_write(data, fname):
    cctx = zstandard.ZstdCompressor(level=17, threads=32)
    with open(fname, 'wb') as f, cctx.stream_writer(f) as z:
        pickle.dump(data, z)

def z_read(fname):
    dctx = zstandard.ZstdDecompressor()
    with open(fname, 'rb') as f, dctx.stream_reader(f) as z:
        return pickle.load(z)

def read_ckpt(ckpt_dir, load_opt=True):
    result = {'params': pmap_on(z_read(ckpt_dir + '/params.pickle.zst'))}
    if load_opt:
        result['opt_state'] = pmap_on(z_read(ckpt_dir + '/opt_state.pickle.zst'))
    return result

def write_ckpt(x, ckpt_dir):
    params, opt_state = map(pmap_off, (x['params'], x['opt_state']))
    z_write(params, ckpt_dir + '/params.pickle.zst')
    z_write(opt_state, ckpt_dir + '/opt_state.pickle.zst')

def read_ckpt(ckpt_dir, load_opt=True):
    result = {'params': pmap_on(pickle.load(open(ckpt_dir + '/params.pickle', 'rb')))}
    if load_opt:
        result['opt_state'] = pmap_on(pickle.load(open(ckpt_dir + '/opt_state.pickle', 'rb')))
    return result

def write_ckpt(x, ckpt_dir):
    params, opt_state = map(pmap_off, (x['params'], x['opt_state']))
    pickle.dump(params, open(ckpt_dir + '/params.pickle', "wb"))
    pickle.dump(opt_state, open(ckpt_dir + '/opt_state.pickle', "wb"))

def save(network, step, bucket, path, aux=None, keep_n=3, delete_old=True):
    assert path
    client = storage.Client()

    if aux is None:
        aux = {}

    try:
        with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
            meta = json.load(f)
    except:
        # create metadata file
        with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
            json.dump({
                "step": 0,
                "checkpoints": [],
                "aux": {}
            }, f)

    # do sharded checkpoint writing
    start = time.time()
    res = []
    write_ckpt(network.state, f"gs://{bucket}/{path}/step_{step}/")

    print(f"Wrote checkpoint in {time.time() - start:.06}s")

    with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
        meta = json.load(f)

    meta["step"] = step
    meta["checkpoints"].append(step)
    all_aux = meta.get("aux", {})

    while len(meta["checkpoints"]) > keep_n:
        ckpt_to_delete = meta["checkpoints"].pop(0)

        try:
            del all_aux[str(ckpt_to_delete)]
        except:
            print(f"failed to delete the aux state for {step}")

        if delete_old:
            print(f"deleting checkpoint {ckpt_to_delete}")
            for blob in client.list_blobs(bucket, prefix=f"{path}/step_{ckpt_to_delete}/"):
                # print(f"deleting {blob.name}")
                assert path in blob.name
                blob.delete()
        else:
            print(f"keeping checkpoint {ckpt_to_delete}")

    all_aux[step] = aux
    meta["aux"] = all_aux

    with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
        json.dump(meta, f)

def gpt3_schedule(warmup_steps,
                  total_steps,
                  peak_lr,
                  end_lr):
    def sch(step):
        warmup_pct = jnp.clip(step, 0, warmup_steps) / warmup_steps
        anneal_pct = jnp.clip(step - warmup_steps, 0, total_steps) / total_steps

        return warmup_pct * peak_lr - (peak_lr - end_lr) * (1 - jnp.cos(jnp.pi * anneal_pct)) / 2

    return sch

context_len = 75

def reshape_data(data):
    # trim data to be multiple of context_len
    chunks = data.shape[-1] // context_len
    data = data[...,:chunks*context_len]
    # bs, chunks*context_len -> bs*chunks, context_len
    data = data.flatten()
    data = data.reshape(-1, context_len)

    # trim data to be multiple of (context_len, context_len)
    chunks = data.shape[0] // context_len
    data = data[:chunks*context_len]
    # chunks*context_len, context_len -> chunks, context_len, context_len
    data = data.reshape((-1, context_len, context_len))
    # make all kind of input length - from 1 to 75 by zeroing the rest
    return np.tril(data)

def add_sot_eot(data):
    sot_token, eot_token = 49406, 49407
    # add sot_token to the start of each sample
    sots = np.full((data.shape[0],context_len,1), sot_token)
    data = np.concatenate((sots, data),axis=-1) 
    # add zeros column at end, so we can add eot_token for the last sample
    zeroes = np.full((data.shape[0],context_len,1), 0)
    data = np.concatenate((data, zeroes),axis=-1) 
    # make diag eot_token matrix
    eots = np.diagflat(np.full(context_len, eot_token))
    # move eot_token two position from diagonal
    zeroes = np.full((context_len,2), 0)
    eots = np.concatenate((zeroes, eots),axis=-1) 
    # place eot_token
    return data + eots

def align_to_devices(data):
    data = data.reshape(-1, data.shape[-1])
    chunks = data.shape[0] // jax.device_count()
    return data[:chunks*jax.device_count()]

def network_step(network, data):
    data = reshape_data(data)    
    orig_data = add_sot_eot(data.copy())
    
    data, orig_data = map(align_to_devices, (data, orig_data))

    inputs = {
        "obs": data,
        "target": orig_data,
    }

    return network.train(inputs)

if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    batch_size = params["batch_size"]

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]

    val_batches = params["val_batches"]
    val_every = params["val_every"]
    ckpt_every = params["ckpt_every"]
    keep_every = params["keep_every"]
    total_steps = params["total_steps"]

    warmup_steps = params["warmup_steps"]
    anneal_steps = params["anneal_steps"]
    lr = params["lr"]
    end_lr = params["end_lr"]
    weight_decay = params["weight_decay"]
   
    # alpha parameter for the exponential moving averages used to compute B_simple
    noise_scale_alpha = params.get("noise_scale_alpha", 0.01)

    scheduler = gpt3_schedule(warmup_steps, anneal_steps, lr, end_lr)
    
    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        optax.clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.add_decayed_weights(weight_decay),
        optax.scale(-1),
        optax.scale_by_schedule(scheduler)
    )

    params["optimizer"] = opt

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    devices = np.array(jax.devices())

    # pick initial ckpt - based on tuning vs train from scratch

    step = 0
    initial_ckpt_state_path = None
    train_loader = None

    if args.tune_model_path:
        print('`--tune_model_path` passed: we are beginning a fine-tuning run')
        fine_tuning = True
        initial_ckpt_state_path = args.tune_model_path
    else:
        print('`--tune_model_path` not passed: we are continuing a fine-tuning run from a checkpoint (or we are not fine-tuning)')
        fine_tuning = False
        initial_ckpt_model_dir = model_dir
        initial_ckpt_path = f"gs://{bucket}/{initial_ckpt_model_dir}"
        meta_path = f"{initial_ckpt_path}/meta.json"

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            ckpt_step = meta["checkpoints"][-1]
            initial_ckpt_state_path = f"{initial_ckpt_path}/step_{ckpt_step}/"
            print(f"state will be restored from checkpoint {ckpt_step}")

            step = ckpt_step
            train_loader = meta['aux'][str(ckpt_step)].get("train_loader", None)
        except NotFound:
            # no checkpoint, start at zero
            print(f"No checkpoint to load at {initial_ckpt_path}. Training from scratch.")

    if initial_ckpt_state_path:
        print(f"path to load checkpoint from: {initial_ckpt_state_path}")
    else:
        print("not loading from a checkpoint")

    # set up datasets
    print("setting up datasets")

    train_dataset = TFRecordNewInputs(f"data/{params['train_set']}",
                                      batch_size=(
                                          gradient_accumulation_steps,
                                          batch_size),
                                      sample_size=2048,
                                      restore_state=train_loader)

    val_sets = {}
    for k, v in params["val_set"].items():
        val_sets[k] = TFRecordNewInputs(
            f"data/{v}", batch_size=(batch_size,), sample_size=2048
        )

    # tok/sec metrics
    sequences_per_step = gradient_accumulation_steps * batch_size * (2048//context_len)

    # load + run
    print("initializing network")
    network = ClipTrainer(params)
    network.state['params'], network.state['opt_state'] = map(pmap_on, (network.state['params'], network.state['opt_state']))

    if initial_ckpt_state_path:
        print("loading network")
        if fine_tuning:
            # get the scheduler step stored in the just-initialized optimizer
            # should be zero
            init_sched_state = network.state["opt_state"]

        start = time.time()
        network.state = read_ckpt(initial_ckpt_state_path, load_opt=(not args.fresh_opt))
        network.state['step'] = ckpt_step
        if fine_tuning:
            # overwrite the loaded scheduler step with zeros
            # this makes fine-tuning use the lr schedule in
            network.state["opt_state"] = init_sched_state

        print(f"network loaded in {time.time() - start:.06}s")

    print('compiling train fn')
    start = time.time()
    loss = network_step(network, train_dataset.get_samples())
    step += 1
    print(f"Train fn compiled in {time.time() - start:.06}s")

    print('compiling eval fn')
    start = time.time()
    for val_set in val_sets.values():
        network_step(network, val_set.get_samples())
        val_set.reset()
    print(f"Eval fn compiled in {time.time() - start:.06}s")
    
    project = params.get("wandb_project", "text-clip")
    wandb.init(project=project, name=params["name"], config=params)
    
    while True:
        if (step % ckpt_every == 1) or step == total_steps:
            print(f"saving a checkpoint for step {step}")
            save(network, step, bucket, model_dir,
                    aux={"train_loader": train_dataset.get_state()},
                    delete_old=True,
                    )

        if step % val_every == 1:  # 1 because we've already taken a step to compile train fn
            for name, val_set in val_sets.items():
                val_loss = []
                for i, _ in tqdm(zip(val_set.sample_once(), range(val_batches)),
                                    desc=f"validation for step {step}, set {name}",
                                    total=val_batches):
                    val_loss.append(network_step(network, i))
                val_set.reset()

                val_loss = np.array(val_loss).mean()
                print(f"validation loss for step {step}, set {name}: {val_loss}")

                #wandb.log({f'val/loss_{name}': float(val_loss)}, step)

        if step == total_steps:
            print("training completed!")
            exit()

        start = time.time()
        loss = network_step(network, train_dataset.get_samples())
        step += 1

        steps_per_sec = 1 / (time.time() - start)
        sequences_processed = sequences_per_step * step

        ### compute summary stats about the gradient

        # converts from grads-summed-over-microbatch (what `CasualTransformer.train` computes)
        # to grads-averaged-over-microbatch (what we want)
        #
        # (when taking gradient steps, the same conversion happens inside the optimizer
        #  via optax.scale(1 / gradient_accumulation_steps))
        
        
        wandb_stats = {
            "train/loss": loss,
            "train/steps_per_sec": steps_per_sec,
            "train/learning_rate": float(scheduler(network.state["opt_state"][-1].count[0].item())),
            "sequences_processed": sequences_processed,
        }
        wandb.log(wandb_stats, step)
        