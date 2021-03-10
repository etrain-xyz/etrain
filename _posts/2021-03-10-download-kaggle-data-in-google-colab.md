---
title: "Tải dữ liệu từ kaggle về Google Colab"
date: 2021-03-10 10:01:00
image: '/assets/img/posts/kaggle-data.png'
description: 'Tải dữ liệu từ kaggle về Google Colab'
main-class: 'guide'
tags:
- python
- kaggle dataset
- google colab
categories:
twitter_text: 'Tải dữ liệu từ kaggle về Google Colab'
introduction: 'Tải dữ liệu từ kaggle về Google Colab'
---

Các bước để tải dữ liệu từ kaggle về Google Colab

### Cài đặt kaggle lên Google Colab

- Đăng nhập và vào mục [Account](https://www.kaggle.com/me/account).
- Kéo xuống dưới phần API và bấm **Create New API Token**. Bạn sẽ tải về một tập tin `kaggle.json` về máy tính.

![Kaggle Account API](/assets/img/posts/kaggle-account-api.png)

- Vào Google Colab và tạo một dự án mới rồi chạy các lệnh sau

```
!pip install -q kaggle

from google.colab import files
files.upload()
# Chọn file kaggle.json tải về ở bước 2 để tải lên

!mkdir '/root/.kaggle'
!cp kaggle.json '/root/.kaggle'
!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets list
# Chạy lệnh liệt kê các bộ dữ liệu của bạn để đảm bảo bạn đã cài đặt thành công
```


<div>
<div class="screen-tv">
<a class="image-link" href="https://pwieu.com/click-FQLMKJP1-KHEQCJKZ?bt=25&tl=1&url=https%3A%2F%2Fshopee.vn%2Fp-i.162160763.7852858918"><img src="/assets/img/ads/Essential-Airfryer-XL-HD9261-philips.gif"></a>
</div>
<img class="cabinet-img" src="/assets/img/cabinet-tv.png">
</div>


### Tải dữ liệu trên kaggle

Bây giờ bạn vào một dataset bất kỳ mà bạn muốn tải về Google Colab. Ở đây Etrain sẽ chọn ví dụ tải về bộ dữ liệu [VinBigData Original Image Dataset](https://www.kaggle.com/awsaf49/vinbigdata-original-image-dataset)

Bạn bấm menu ở bên góc phải và chọn **Copy API command**

![Kaggle Dataset API Command](/assets/img/posts/kaggle-data-api-cmnd.png)


Sau đó dán vào Google Colab để chạy lệnh. Lệnh sau khi sao chép với bộ dữ liệu trên sẽ như sau

```
!kaggle datasets download -d awsaf49/vinbigdata-original-image-dataset
```

Bây giờ bạn có thể giải nén và sử dụng bộ dữ liệu

```
!unzip -q vinbigdata-original-image-dataset.zip
```

### Nguồn tham khảo
[Easiest way to download kaggle data in Google Colab](https://www.kaggle.com/general/74235)