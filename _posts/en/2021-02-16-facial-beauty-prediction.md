---
title: "Facial beauty prediction"
date: 2021-02-16 23:16:23
image: '/assets/img/posts/SCUT-FBP5500.jpg'
description: 'Face beauty score model'
main-class: 'demo'
tags:
- python
- deep learning
- face recognition
categories:
twitter_text: 'Face beauty score model'
introduction: 'Face beauty score model'
---

## Introduction

This article introduce the method of scoring facial beauty using deep learning

### Database

The dataset is SCUT-FBP5500, it has totally 5500 frontal faces. The SCUT-FBP5500 Dataset can be divided into four subsets with different races and gender, including 2000 Asian females(AF), 2000 Asian males(AM), 750 Caucasian females(CF) and 750 Caucasian males(CM).

![SCUT-FBP5500](/assets/img/posts/SCUT-FBP5500.jpg)

All the images are labeled with beauty scores ranging from [1, 5] by totally 60 volunteers.

![SCUT-FBP5500-score](/assets/img/posts/SCUT-FBP5500-score.png)

For more detail you can see the [SCUT-FBP: A Benchmark Datasetfor Facial Beauty Perception](https://arxiv.org/pdf/1511.02459.pdf) paper.

### Training and Testing Set Split

We use two kinds of experimental settings to evaluate the facial beauty prediction methods on SCUT-FBP5500 benchmark, which includes:

1. 5-folds cross validation. For each validation, 80% samples (4400 images) are used for training and the rest (1100 images) are used for testing.
2. The split of 60% training and 40% testing. 60% samples (3300 images) are used for training and the rest (2200 images) are used for testing.

### Benchmark Evaluation

We set ResNet-18, ResNet-50 and ResNeXt-50 as the benchmarks of the SCUT-FBP5500 dataset, and we evaluate the benchmark on various measurement metrics, including: Pearson correlation (PC), maximum absolute error (MAE), and root mean square error (RMSE).

![SCUT-FBP5500-score](/assets/img/posts/SCUT-FBP5500-Benchmark.png)


## Training Tutorials and Models

### Install

Create new [Google Colab](https://colab.research.google.com) notebook, pull repo and install dependencies

```
%cd /content
!git clone https://github.com/etrain-xyz/facial-beauty-prediction.git
%cd /content/facial-beauty-prediction
!pip install -r requirements.txt
```

Download the [SCUT-FBP5500](https://drive.google.com/open?id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf) dataset and unzip.

```
!pip install gdown
!gdown https://drive.google.com/uc?id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf
!unzip -q 'SCUT-FBP5500_v2.1.zip'
```

We will train with `resnet18`, you can see [model.py](https://github.com/etrain-xyz/facial-beauty-prediction/blob/master/model.py) file for more detail.

```python
#config.py
import torch
import torchvision.transforms as transforms

model_arch = 'resnet18'
epochs = 50

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

transform = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

data_root = './SCUT-FBP5500_v2'

models_dir = './models'
```

### Training

Training by 60% training and 40% testing.

```python
#train.py
...
if __name__ == '__main__':
	# The split of 60% training and 40% testing
	train_dir = os.path.join(config.data_root, 'train_test_files/split_of_60%training and 40%testing/train.txt')
	val_dir = os.path.join(config.data_root, 'train_test_files/split_of_60%training and 40%testing/test.txt')
	saved_path = os.path.join(config.models_dir, config.model_arch+'_best_state.pt')
	train(train_dir, val_dir, model_saved_path=saved_path)
```

Training

```
!mkdir /content/facial-beauty-prediction/models
!python train.py
```

### Testing

Download the `5_face_landmarks` of dlib and unzip.

```
%cd /content/facial-beauty-prediction/models
!wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
!bzip2 -d shape_predictor_5_face_landmarks.dat.bz2
```

Run predict

```
%cd /content/facial-beauty-prediction/
!python predict.py -i ./SCUT-FBP5500_v2/Images/AF883.jpg -m ./models/resnet18_best_state.pt
```

Show some results

```python
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

images = []
for img_path in glob.glob('./SCUT-FBP5500_v2/Images/result_*'):
    images.append(mpimg.imread(img_path))

plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(image)
```

![SCUT-FBP5500-score](/assets/img/posts/SCUT_FBP5500-result.png)

## References

Click [here]((https://github.com/etrain-xyz/facial-beauty-prediction)) to see more