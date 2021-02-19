---
title: "Đánh giá vẻ đẹp khuôn mặt"
date: 2021-02-16 23:16:23
image: '/assets/img/posts/SCUT-FBP5500.jpg'
description: 'Mô hình chấm điểm khuôn mặt.'
main-class: 'demo'
tags:
- python
- deep learning
- face recognition
categories:
twitter_text: 'Mô hình chấm điểm khuôn mặt.'
introduction: 'Mô hình chấm điểm khuôn mặt.'
---

## Giới thiệu

Bài viết này giới thiệu phương pháp đánh giá vẻ đẹp khuôn mặt bằng cách sử dụng mô hình học sâu trong lĩnh vực trí tuệ nhân tạo.

### Dữ liệu

Dữ liệu sử dụng để huấn luyện là bộ SCUT-FBP5500. Bộ dữ liệu có tổng số 5500 khuôn mặt trực diện bao gồm 4 tập con với các chủng tộc và giới tính khác nhau, bao gồm 2000 ảnh khuôn mặt phụ nữ Châu Á (AF), 2000 ảnh khuôn mặt đàn ông Châu Á (AM), 750 khuôn mặt phụ nữ da trắng, 750 khuôn mặt đàn ông da trắng (CM).

![SCUT-FBP5500](/assets/img/posts/SCUT-FBP5500.jpg)

Các ảnh được gán nhãn với điểm số sắc đẹp khác nhau theo thang điểm từ 1 đến 5 bởi 60 tình nguyện viên.

![SCUT-FBP5500-score](/assets/img/posts/SCUT-FBP5500-score.png)

Các bạn có thể xem thêm chi tiết trong bài báo [SCUT-FBP: A Benchmark Datasetfor Facial Beauty Perception](https://arxiv.org/pdf/1511.02459.pdf)

### Tập huấn luyện và thử nghiệm

Sử dụng 2 cách chia dữ liệu khác nhau để đánh giá mô hình

1. 5-folds cross validation. Mỗi fold được chia theo tỷ lệ 80% mẫu (4400 ảnh) huấn luyện và 20% mẫu (1100 ảnh) cho việc thử nghiệm.
2. Chia tập dữ liệu theo tỷ lệ 60% mẫu (3300 ảnh) huấn luyện và 40% mẫu (2200 ảnh) cho việc thử nghiệm.

### Đánh giá mô hình

Etrain sử dụng 3 mô hình khác nhau trong `torchvision.models` là ResNet-18, ResNet-50, ResNeXt-50 và cũng đánh giá mô hình theo 3 chỉ số: PC, MAE, và RMSE.

![SCUT-FBP5500-score](/assets/img/posts/SCUT-FBP5500-Benchmark.png)


## Huấn luyện mô hình đánh giá

### Cài đặt

Các bạn vào google colab để tạo notebook [Google Colab](https://colab.research.google.com). Tải repo và cài đặt các gói cần thiết

```
%cd /content
!git clone https://github.com/etrain-xyz/facial-beauty-prediction.git
%cd /content/facial-beauty-prediction
!pip install -r requirements.txt
```

Tải bộ dữ liệu [SCUT-FBP5500](https://drive.google.com/open?id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf) và giải nén.

```
!pip install gdown
!gdown https://drive.google.com/uc?id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf
!unzip -q 'SCUT-FBP5500_v2.1.zip'
```

<div>
<div class="screen-tv">
<a class="image-link" href="https://pinggo.vn/products/Xiaomi-May-hut-am-New-Widetech-2-dung-tich?ref=75576dd5c4"><img src="/assets/img/ads/xiaomi-new-widetech.gif"></a>
</div>
<img class="cabinet-img" src="/assets/img/cabinet-tv.png">
</div>


Trong bài viết này Etrain sẽ train dữ liệu với `resnet18`, bạn có thể tìm hiểu thêm ở file [model.py](https://github.com/etrain-xyz/facial-beauty-prediction/blob/master/model.py).

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

### Huấn luyện

Sử dụng bộ dữ liệu 60% huấn luyện và 40% thử nghiệm

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

Huấn luyện dữ liệu

```
!mkdir /content/facial-beauty-prediction/models
!python train.py
```

### Thử nghiệm

Tải `5_face_landmarks` của thư viện dlib

```
%cd /content/facial-beauty-prediction/models
!wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
!bzip2 -d shape_predictor_5_face_landmarks.dat.bz2
```

Chạy thử nghiệm

```
%cd /content/facial-beauty-prediction/
!python predict.py -i ./SCUT-FBP5500_v2/Images/AF883.jpg -m ./models/resnet18_best_state.pt
```

Xem kết quả một số hình ảnh

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
