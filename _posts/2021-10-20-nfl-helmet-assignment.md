---
title: "Gán nhãn mũ bảo hiểm người chơi bóng bầu dục"
date: 2021-10-20 14:33:00
image: '/assets/img/posts/nfl-helmet-assignment.jpg'
description: 'Gán nhãn mũ bảo hiểm người chơi bóng bầu dục'
main-class: 'guide'
tags:
- python
- deep learning
- object detection
- multi-object tracking
categories:
twitter_text: 'Gán nhãn mũ bảo hiểm sử dụng Yolov5 Deepsort Pytorch'
introduction: 'Gán nhãn mũ bảo hiểm sử dụng Yolov5 Deepsort Pytorch'
---

### Giới thiệu
[Cuộc thi "2021 NFL Health & Safety, Helmet Assignment"](https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment) là một phần trong nỗ lực hợp tác giữa National Football League (NFL) và Amazon Web Services (AWS) nhằm đưa ra một chương trình giám sát và giảm thiểu chấn thương thể thao tốt nhất. Chúng tôi đang cố gắng đánh nhãn đúng người chơi dựa trên việc phát hiện và theo dõi mũ bảo hiểm của người đó thông qua các đoạn video. Các nhãn bao gồm chữ H (ký hiệu dành cho đội chủ nhà) và chữ V (ký hiệu dành cho đội khách) và theo sau là số áo của cầu thủ. Nhãn người chơi được cung cấp trong dữ liệu theo dõi NGS (Next Gen Stats) cho mỗi lần chơi. Một bài gửi hoàn hảo sẽ xác định chính xác vị trí của mũ bảo hiểm, gán nhãn mũ bảo hiểm đúng người chơi trên mọi khung hình của video.

<div>
<div class="screen-tv">
<a class="image-link" href="https://pwieu.com/v2/click-bOPZ7-xdG1Kp-MjDGZ-0b579e72?tl=1&url=https%3A%2F%2Fshopee.vn%2Fp-i.299252.7841386023"><img src="/assets/img/ads/xiaomi-tv-stick.gif"></a>
</div>
<img class="cabinet-img" src="/assets/img/cabinet-tv.png">
</div>

### Giải pháp
Cách tiếp cận rất đơn giản:
1. [Áp dụng thuật toán deepsort cho mỗi khung hình](https://www.kaggle.com/duythanhng/nfl-helmet-with-yolov5-deepsort-starter). Việc này sẽ phân cụm mũ bảo hiểm trên mỗi khung hình khi nó là của cùng một người chơi/mũ bảo hiểm
2. [Gom nhóm những cụm deepsort này và chọn nhãn phổ biến nhất cho cụm đó](https://www.kaggle.com/duythanhng/nfl-yolov5-deepsort-pytorch-guide). Sau đó đánh nhãn tất cả các mũ bảo hiểm đó (phát hiện mũ bảo hiểm dùng yolov5) cho cùng một người chơi.

<iframe width="560" height="315" src="https://www.youtube.com/embed/TofMADTFkjI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Tài liệu tham khảo
[NFL Helmet Assignment - Getting Started Guide](https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide)

[Helper Code + Helmet Mapping + Deepsort](https://www.kaggle.com/robikscube/helper-code-helmet-mapping-deepsort)

[NFL helmet with Yolov5-deepsort starter](https://www.kaggle.com/s903124/nfl-helmet-with-yolov5-deepsort-starter)