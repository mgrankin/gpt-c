# gpt-c
# GPT-C - a GPT trained with additional CLIP supervision. 
We assume a lots of visual information. These assumption are crucial to text understanding but they appear in text quite rare. The CLIP network was trained to produce similar vectors from text for things that do have visual similarity. CLIP network has been trained that “rose” is not that far from “pink” and that “square” is a case of “rectangle” and many more. The idea is to add additional CLIP supervision in GPT training - the GPT-C model.

# Upgrading CLIP
Text CLIP is a transformer with triangular attention matrix. It makes N vectors for sequence lenght of N tokens. Because of that, similar to GPT, K-th output only depends on K first tokens. It’s almost what we need to add to GPT. There are two obstacles. First - positional encoding, in CLIP it’s absolute positional encoding. This is not flexible for our purpose, RoPE would be great. Second - special “start of text”, “end of text” tokens. The CLIP network was trained to make relevant output only in place of the special “end of text" token. We need it to make sense for each output. I trained the upgraded text CLIP using the original CLIP for supervision. It trained for 350k steps till 0.007 MSE loss on validation set.

### The Results

All models were trained for 350k steps.
Validation loss are 
Plain GPT with original tokenizer - 2.301
Plain GPT with tokenizer from CLIP - 2.39
GPT-C model - 

wandb runs https://wandb.ai/grankin/gpt-c

### Acknowledgements

The research was made possible with TRC program from Google. The code for CLIP and GPT is from kingoflolz's repos.

### How to replicate

```bash
cd gpt-c
mamba env create
conda activate gpt-c
```

Download The Pile from here https://the-eye.eu/public/AI/pile/

```bash
wget -m -np -c -U "eye02" -w 2 -R "index.html*" "https://the-eye.eu/public/AI/pile/"
```

Tokenize the dataset using CLIP's tokenizer with tokenization code from gpt-neo
```bash
wget https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz -O bpe_simple_vocab_16e6.txt.gz
PILE=/home/u/data/the-eye.eu/public/AI/pile
TOKENS=/home/u/data/pile
python create_tfrecords.py --input_dir $PILE --name pile --files_per 300000 --output_dir $TOKENS --write_dataset_config --processes 2
cd $TOKENS
mv pile_0_164596.tfrecords pile_test.tfrecords
mv pile_0_169067.tfrecords pile_valid.tfrecords
mkdir train
python create_tfrecords.py --input_dir $PILE/train --name pile --files_per 300000 --output_dir $TOKENS/train --write_dataset_config --processes 5
```

Copy the data to your bucket

```bash
sudo snap install google-cloud-sdk --classic
gcloud init
gsutil mb gs://mgrankin
gsutil -m cp -r $TOKENS gs://mgrankin/datasets/pile_clip
mkdir data
cd data
echo "gs://mgrankin/datasets/pile_clip/pile_valid.tfrecords" > pile_clip.val.index
gsutil ls gs://mgrankin/datasets/pile_clip/train > pile_clip.train.index
```

Create and connect to tpu

```bash
gcloud alpha compute tpus tpu-vm delete tpu1

gcloud alpha compute tpus tpu-vm create tpustart_1 \
--zone us-central1-a \
--accelerator-type v3-8 \
--version v2-alpha

gcloud alpha compute tpus tpu-vm ssh tpustart_1 --zone us-central1-a
```

Inside the tpu

Small touch of convinience
```bash
yes | sudo apt install tmux zsh
#
curl -L http://install.ohmyz.sh | sh
echo "ZSH_DISABLE_COMPFIX=true
$(cat .zshrc)" > .zshrc
#
cd 
git clone https://github.com/gpakosz/.tmux.git
ln -s -f .tmux/.tmux.conf
cp .tmux/.tmux.conf.local .
echo "set-window-option -g mouse on" >>  .tmux.conf.local
echo "set-option -g default-shell /bin/zsh" >>  .tmux.conf.local

```

Run learning inside tmux, wont crash if ssh fail
```bash
tmux attach-session -t gpt || tmux new-session -s gpt /bin/zsh 

git config --global credential.helper store
git clone https://github.com/mgrankin/gpt-c.git
#
git clone https://github.com/mgrankin/mesh-transformer-jax-clip
cp gpt-c/data/* mesh-transformer-jax-clip/data
cd mesh-transformer-jax-clip
pip install --upgrade pip
pip install -r requirements.txt
pip install "jax[tpu]>=0.2.19" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Train baseline models
```bash
cd 
cd mesh-transformer-jax-clip
python3 device_train.py --config=../gpt-c/configs/162M_roto_8.json 
python3 device_train.py --config=../gpt-c/configs/162M_baseline.json
```

Train text CLIP
```bash
cd
cd gpt-c/
python3 clip_device_train.py --config=./configs/text_clip_lr.json
```

Train the main model
```bash
cd 
cd gpt-c
pip install .
cd
cd mesh-transformer-jax-clip
git checkout start
mkdir model
cd model
gsutil -m cp -r gs://mgrankin/text_clip_r16_lr10/step_350000//params.pickle text_clip.pickle
cd ..
python3 device_train.py --config=../gpt-c/configs/162M_start.json 
```
