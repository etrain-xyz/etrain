---
title: "Self-supervised speech recognition"
date: 2021-01-15 15:43:23
image: '/assets/img/posts/SpeechToText-1024x473.png'
description: 'Self-supervised speech recognition with limited amount of labeled data'
main-class: 'demo'
color: '#22857b'
tags:
- python
- NLP
- deep learning
- self supervised
- speed recognition
categories:
twitter_text: 'Self-supervised speech recognition with limited amount of labeled data'
introduction: 'Self-supervised speech recognition with limited amount of labeled data'
---

## What is Self-supervised learning

First, we need to know about the self-supervised learning, what problem does it solve.

As manufacturers begin to integrate AI solutions into production lines, data scarcity has emerged as a major challenge. Unlike consumer Internet companies, which have data from billions of users to train powerful AI models, collecting massive training sets in manufacturing is often not feasible.

For example, in automotive manufacturing, where lean Six Sigma practices have been widely adopted, most OEMs and Tier One suppliers strive to have fewer than three to four defects per million parts. The rarity of these defects makes it challenging to have sufficient defect data to train visual inspection models.

![robot](/assets/img/posts/xrobot_with_books-amatech.png)

Big data has enabled AI in consumer internet companies. Can manufacturing also make AI work with small data? In fact, recent advances in AI are making this possible.

The  to circumvent the small data problem to help their AI projects go live even with only dozens or fewer examples. Self-supervised learning is one of the techniques that circumvent the small data problem to help their AI projects go live even with only dozens or fewer examples.

Self-supervised learning is similar to transfer-learning, but the obtained knowledge is acquired by solving a slightly different task and then adapted to small data problem. For example, you can take a lot of OK images and create a puzzle-like grid to be sorted by a base model. Solving this dummy problem will force the model to acquire domain knowledge that can be used as starting point in the small data task.

## Using for the Vietnamese speech to text problem

["Self-supervised speech recognition with limited amount of labeled data"](https://github.com/mailong25/self-supervised-speech-recognition) is a repo using the self-supervised learning for the speech to text problem. We will test with the Vietnamese pretrained-model.

![speech to text](/assets/img/posts/SpeechToText-1024x473.png)

You access [Google Colab](https://colab.research.google.com) and create new notebook. Download the pretrained-model at [google drive](https://drive.google.com/file/d/1kZFdvMQt-R7fVebTbfWMk8Op7I9d24so/view?usp=sharing).


```bash
# Install gdown
!pip install gdown
# Download pretrained model
%cd /content
!gdown https://drive.google.com/uc?id=1kZFdvMQt-R7fVebTbfWMk8Op7I9d24so
# Unzip
!unzip -q vietnamese_wav2vec.zip
```

Clone the Etrain repo. This repo is forked from the original repo. The goal was to ensure the demo running on Colab.


```bash
%cd /content
!git clone https://github.com/etrain-xyz/self-supervised-speech-recognition.git
# Install soundfile lib
!pip install soundfile
```

Check cuda version

```bash
!nvcc --version
#nvcc: NVIDIA (R) Cuda compiler driver
#Copyright (c) 2005-2019 NVIDIA Corporation
#Built on Sun_Jul_28_19:07:16_PDT_2019
#Cuda compilation tools, release 10.1, V10.1.243
```

Install `torch`

```bash
!pip install torchvision==0.7.0
# If cuda version < 11 (eg. cuda 10.1):
!pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# If cuda >= 11:
!pip install torch==1.6.0
```

Install `fairseq`
```bash
!pip install fairseq==0.9.0
!pip show fairseq
%cd /content/
!git clone https://github.com/pytorch/fairseq.git
%cd fairseq
!git checkout c8a0659be5cdc15caa102d5bbf72b872567c4859
!pip install --editable ./
!python setup.py build develop
```

Install dependencies for `wav2letter`

```bash
!apt-get update && apt-get upgrade -y && apt-get install -y && apt-get -y install apt-utils gcc libpq-dev libsndfile-dev
!apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
!apt-get install libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev
```

Install `kenlm`

```bash
%cd /content/
!git clone https://github.com/kpu/kenlm.git
%cd /content/kenlm/
!mkdir -p build
%cd build
!cmake ..
!make -j 4
```

Install `wav2letter`

```bash
%cd /content/
!rm -rf /content/wav2letter
!git clone -b v0.2 https://github.com/facebookresearch/wav2letter.git
%cd wav2letter/bindings/python
!export KENLM_ROOT_DIR=/content/kenlm/ && pip install -e .
```

We use the `Common Voice Corpus 5.1` dataset in [Common Voice](https://commonvoice.mozilla.org/vi/datasets) for this demo. Download the dataset and upload to your drive and mount the directory in drive to colab.

```bash
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
```

The dataset file path is `"/content/drive/My Drive/Etrain/speech-to-text/vi.tar.gz"`. Unzip file to Colab and convert the audio data format from mp3 to wav using `ffmpeg`.


```bash
%cd /content/
!tar -xzf '/content/drive/My Drive/Etrain/speech-to-text/vi.tar.gz'
# Make the directory containing wav audio
!mkdir /content/data
# We test with 5 audio files
%cd /content/cv-corpus-5.1-2020-06-22/vi/clips
!ffmpeg -i common_voice_vi_21833280.mp3 -acodec pcm_s16le -ac 1 -ar 16000 /content/data/common_voice_vi_21833280.wav
!ffmpeg -i common_voice_vi_21833328.mp3 -acodec pcm_s16le -ac 1 -ar 16000 /content/data/common_voice_vi_21833328.wav
!ffmpeg -i common_voice_vi_21861994.mp3 -acodec pcm_s16le -ac 1 -ar 16000 /content/data/common_voice_vi_21861994.wav
!ffmpeg -i common_voice_vi_21964746.mp3 -acodec pcm_s16le -ac 1 -ar 16000 /content/data/common_voice_vi_21964746.wav
!ffmpeg -i common_voice_vi_22008973.mp3 -acodec pcm_s16le -ac 1 -ar 16000 /content/data/common_voice_vi_22008973.wav
```

Make `config.txt` to run demo

```bash
!rm -f /content/self-supervised-speech-recognition/config.txt
!echo "[TRANSCRIBER]" >> /content/self-supervised-speech-recognition/config.txt
!echo "wav_folder = /content/data" >> /content/self-supervised-speech-recognition/config.txt
!echo "pretrain_model = /content/vietnamese_wav2vec/pretrain.pt" >> /content/self-supervised-speech-recognition/config.txt
!echo "finetune_model = /content/vietnamese_wav2vec/finetune.pt" >> /content/self-supervised-speech-recognition/config.txt
!echo "dictionary = /content/vietnamese_wav2vec/dict.ltr.txt" >> /content/self-supervised-speech-recognition/config.txt
!echo "lm_lexicon = /content/vietnamese_wav2vec/lexicon.txt" >> /content/self-supervised-speech-recognition/config.txt
!echo "lm_model = /content/vietnamese_wav2vec/lm.bin" >> /content/self-supervised-speech-recognition/config.txt
!echo "lm_weight=1.5" >> /content/self-supervised-speech-recognition/config.txt
!echo "word_score=-1" >> /content/self-supervised-speech-recognition/config.txt
!echo "beam_size=50" >> /content/self-supervised-speech-recognition/config.txt
```

The `config.txt` file

```
[TRANSCRIBER]
wav_folder = /content/data
pretrain_model = /content/vietnamese_wav2vec/pretrain.pt
finetune_model = /content/vietnamese_wav2vec/finetune.pt
dictionary = /content/vietnamese_wav2vec/dict.ltr.txt
lm_lexicon = /content/vietnamese_wav2vec/lexicon.txt
lm_model = /content/vietnamese_wav2vec/lm.bin
lm_weight=1.5
word_score=-1
beam_size=50
```

Run the inference and see the results

```bash
%cd /content/self-supervised-speech-recognition/
!python inference.py
```

### The results

#### 1 - common_voice_vi_21861994.wav
- Content: ĐỨNG TRƯỚC BÀN THỜ TRINH VUI SƯỚNG NÓI
- Prediction: ĐỨNG TRƯỚC BÀN THỜ **CHINH** VUI **TƯỚNG** NÓI

#### 2 - common_voice_vi_21833280.wav
- Content: TAY LÀM HÀM NHAI TAY QUAI MIỆNG TRỄ
- Prediction: **TAI** LÀM HÀM NHAI TAY QUAI MIỆNG TRỄ

#### 3 - common_voice_vi_21833328.wav
- Content: QUAY LẠI SAU CẬU BẠN NHÌN CHINH RỒI NÓI
- Prediction: QUAY LẠI SAU CẬU BẠN NHÌN **TRINH** RỒI NÓI

#### 4 - common_voice_vi_22008973.wav
- Content: THÌ CHẮC CHẮN CON GÁI NHÀ CÔ SẼ GẶP VẬN LỚN
- Prediction: **KHÌ** CHẮC CHẮN CON GÁI NHÀ CÔ SẼ GẮP **VẬT** LỚN

#### 5 - common_voice_vi_21964746.wav
- Content: NHỮNG CÂU CHUYỆN HAY NHỮNG BỨC ẢNH ĐƯỢC CHÚNG TÔI KHOE LẠI
- Prediction: NHỮNG CÂU CHUYỆN **THAY** NHỮNG BỨC ẢNH ĐƯỢC CHÚNG TÔI KHOE **LOẠI**


## References
- [Making AI Work with Small Data](https://www.industryweek.com/technology-and-iiot/digital-tools/article/21122846/making-ai-work-with-small-data)
- [Self-supervised speech recognition with limited amount of labeled data](https://github.com/mailong25/self-supervised-speech-recognition)

