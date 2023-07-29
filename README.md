# To Spice Images

This repository describes the steps to fully synthesize a dataset of  labeled spice images, with precise ground truth bounding boxes. First, synthetic images are generated using a fork of Dall-E Flow, then text is added using a fork of SynthText. The included Jupyter notebook runs tests of text recognition on the images. 

A pre-generated dataset is publicly hosted on AWS, [for a demo in the jupyter notebook](#visualize). This notebook uses EasyOCR on the synthetic images, measuring text prediction accuracy and the precision of the bounding boxes. The Jupyter notebook concludes with a demo of recipe geneation by passing the recognized spices to OpenAI. To reproduce this work, you must have your own OpenAI API key.

## Dall-E Server

Launch `1x A10 (24 GB PCIe)` instance [with Lambda Labs](https://cloud.lambdalabs.com/instances), then ssh and run:


```
sudo chown ubuntu:docker /var/run/docker.sock

sudo apt-get update
sudo apt-get install make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
curl https://pyenv.run | bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
exec $SHELL
pyenv install 3.10.12
pyenv global 3.10.12
pip install -U pip
mkdir dalle && cd dalle
git clone https://github.com/thejohnhoffer/dalle-flow.git
git clone https://github.com/jina-ai/SwinIR.git
git clone --branch v0.0.15 https://github.com/AmericanPresidentJimmyCarter/stable-diffusion.git
git clone https://github.com/CompVis/latent-diffusion.git
git clone https://github.com/jina-ai/glid-3-xl.git
git clone https://github.com/timojl/clipseg.git
python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
python3 -m pip install numpy tqdm pytorch_lightning einops numpy omegaconf
python3 -m pip install https://github.com/crowsonkb/k-diffusion/archive/master.zip
python3 -m pip install git+https://github.com/AmericanPresidentJimmyCarter/stable-diffusion.git@v0.0.15
python3 -m pip install basicsr facexlib gfpgan
python3 -m pip install realesrgan
python3 -m pip install xformers
cd latent-diffusion && python3 -m pip install -e . && cd -
cd stable-diffusion && python3 -m pip install -e . && cd -
cd SwinIR && python3 -m pip install -e . && cd -
cd glid-3-xl && python3 -m pip install -e . && cd -
cd clipseg && python3 -m pip install -e . && cd -
cd glid-3-xl
wget https://dall-3.com/models/glid-3-xl/bert.pt
wget https://dall-3.com/models/glid-3-xl/kl-f8.pt
wget https://dall-3.com/models/glid-3-xl/finetune.pt
cd -
cd dalle-flow
python3 -m pip install Cython
python3 -m pip install -r requirements.txt
python3 -m pip install -U jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python3 -m pip install pytorch-lightning==v1.7.7
python3 -m pip install transformers==4.25.1
python3 -m pip install torchmetrics==0.11.4
cd -
export JINA_AUTH_TOKEN="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"
mkdir ./stable-diffusion/models/ldm/stable-diffusion-v1
wget -O ./stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt
cd dalle-flow
pip install orbax-checkpoint orbax-export
pip install grpcio==1.48.2
pip install jina==3.11.2
pip install dalle_mini
pip install timm
python3 flow_parser.py --enable-stable-diffusion --enable-clipseg
python3 -m jina flow --uses flow.tmp.yml

```

## Dall-E Client

In a separate shell on the same instance, run:

```
python client.py DALLE_PORT DIFFUSION_PORT UPSCALE_PORT SEGMENT_PORT
```

where DALLE_PORT, DIFFUSION_PORT, UPSCALE_PORT, SEGMENT_PORT are from the jina log:

```
    INFO   dalle/rep-0@180585 start server bound to 0.0.0.0:56768
    ...
    INFO   diffusion/rep-0@305004 start server bound to 0.0.0.0:55470
    ...
    INFO   upscaler/rep-0@178370 start server bound to 0.0.0.0:62240
    ...
    INFO   clipseg/rep-0@180633 start server bound to 0.0.0.0:57042
```

That is, you must manually find the ports searching through the server logs from the previous shell. The output will be a directory that depends on the parameters in the `client.py` file. The directory will have this format: `spices-v-3-3-0-check-10-of-12`, where `3-3-0` represents version `3.3.0`, and the "check 10 of 12" parameter describes the process of sorting `12` DALLEE images for prompt matching, thenn comparing the top `10` to an ideal size of the text destination segment.

## Text generation

Run the following in a shell with access to a `spices-v-3-3-0-...etc` folder from the last step.

```
git clone https://github.com/hssrobotics23/SynthText.git
cd SynthText
pyenv install 3.6.7
pyenv local 3.6.7
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install opencv-python
python gen.py "/path/to/spices-v-..."
```

Move the "results" directory to a folder formatted like `results-v-3-3-0-check-10-of-25`. When you have many such folders, run:

```
python merge_results.py
```

Note, the `PROMPTS` object must be updated in `merge_prompts.py` with the prompts for each version of `client.py`.

After running `merge_results.py`, you can sync up with the remote `S3` bucket

```
aws s3 sync merged s3://dgmd-s17-assets/train/generated-text-images/
```

Note, you must have access with `aws configure`. You will also recieve a `merged.json` file, which you can use for the next step


## Visualize

Using [conda](https://docs.anaconda.com/anaconda/install/windows/) is easier!

```
conda create -n visualize python=3.9
conda activate visualize
pip install matplotlib numpy openai
pip install jupyterlab opencv-python
pip install git+https://github.com/JaidedAI/EasyOCR.git@f947eaa36a55adb306feac58966378e01cc67f85
python3 -m pip install --force-reinstall -v "Pillow==9.5.0"
conda install nb_conda_kernels
python3 -m jupyterlab
```

Here are the `pyenv` instructions:

```
pyenv install 3.9.17
pyenv local 3.9.17
python3 -m pip install jupyterlab numpy
python3 -m pip install opencv-python
python3 -m pip install matplotlib openai
python3 -m pip install git+https://github.com/JaidedAI/EasyOCR.git@f947eaa36a55adb306feac58966378e01cc67f85
python3 -m pip install --force-reinstall -v "Pillow==9.5.0"
python3 -m jupyterlab
```

### Running the full notebook

To run the notebook to completion, you'll need:

- a local copy of `merged.json`, with an index to the `s3` demo images
- an openai API key, [available when logged in here](https://platform.openai.com/account/api-keys)

Note, you may follow [these steps](https://albertauyeung.github.io/2020/08/17/pyenv-jupyter.html/) to enable `pyenv` within `jupyterlab`. Note, the git install of EasyOCR is needed until the resolution of [this EasyOCR Issue](https://github.com/JaidedAI/EasyOCR/issues/1077)
