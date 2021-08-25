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
gcloud alpha compute tpus tpu-vm delete tpu3

gcloud alpha compute tpus tpu-vm create tpu3 \
--zone us-central1-a \
--accelerator-type v3-8 \
--version v2-alpha

gcloud alpha compute tpus tpu-vm ssh tpu3 --zone us-central1-a
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
```
tmux attach-session -t gpt || tmux new-session -s gpt /bin/zsh 

git config --global credential.helper store
git clone https://github.com/mgrankin/gpt-c.git
#
git clone https://github.com/kingoflolz/mesh-transformer-jax.git
cp gpt-c/data/* mesh-transformer-jax/data
cd mesh-transformer-jax
pip install -r requirements.txt

pip install jax==0.2.19 jaxlib==0.1.69

pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

Train baseline models
```
cd 
cd mesh-transformer-jax
python3 device_train.py --config=../gpt-c/configs/162M_roto_8.json 
python3 device_train.py --config=../gpt-c/configs/162M_baseline.json
```

Train text CLIP
```
cd
cd gpt-c/
python3 clip_device_train.py --config=./configs/text_clip.json
```
