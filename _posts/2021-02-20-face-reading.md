---
title: "Đoán tính cách qua khuôn mặt"
date: 2021-02-16 23:16:23
image: '/assets/img/posts/face-reading.png'
description: 'Xem tướng sử dụng học sâu.'
main-class: 'funny'
tags:
- python
- deep learning
- face recognition
categories:
twitter_text: 'Xem tướng sử dụng học sâu.'
introduction: 'Xem tướng sử dụng học sâu.'
---

Bài lần trước Etrain đã giới thiệu bài viết [đánh giá vẻ đẹp khuôn mặt](/posts/facial-beauty-prediction), bài viết này Etrain sẽ giới thiệu dự án [face2fate](https://github.com/YeyunLU/Face2Fate) để chúng ta có thể làm một trang web xem tướng qua khuôn mặt cơ bản.

![Face Reading](/assets/img/posts/face-reading.png)

Cấu trúc thư mục:

![Face Reading Project Structure](/assets/img/posts/face-reading-project-sctructure.png)


### Thư mục data

Tải về  2 tập tin `shape_predictor_5_face_landmarks.dat.bz2` và `shape_predictor_68_face_landmarks.dat.bz2` vào thư mục data

```
cd data
wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
bzip2 -d shape_predictor_5_face_landmarks.dat.bz2
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

Tạo tập tin `analysis.json` trong thư mục data

```json
{
  "face_regions":
  [
  {
    "name":"eyebrows",
    "features":[
    {
      "name":"Straight",
      "label": "Lông mày ngang",
      "analysis":"Người này có sự kiên trì và dũng cảm tuyệt vời. Người có lông mày ngang rậm thường là người có ý chí mạnh mẽ, can đảm và nghiêm khắc. Người có lông mày ngang mỏng thường là người thông minh, tài giỏi và sắc sảo."
    },
    {
      "name":"Arch",
      "label": "Lông mày vòng cung",
      "analysis":"Năng nổ, không ngại khó, dũng cảm, tự tin đổi mới. Nhưng đôi khi hơi cứng đầu."
    },
    {
      "name":"Circle",
      "label": "Lông mày tròn",
      "analysis":"Nhẹ nhàng, thân thiện và chu đáo, thường có mối quan hệ tốt với người khác. Có cảm xúc mạnh mẽ với nghệ thuật, nhưng đôi khi cũng có cảm xúc và lý tưởng trong cuộc sống đời thường."
    }
    ]
  },
  {
    "name":"eyes",
    "features":[
    {
      "name":"Small",
      "label": "Mắt nhỏ",
      "analysis":"Lý trí, thường phù hợp với nghiên cứu khoa học. Có chủ kiến ​​của riêng mình, không dễ bị ảnh hưởng bởi người khác."
    },
    {
      "name":"Slit",
      "label": "Mắt híp",
      "analysis":"Thường có ngoại hình đẹp chỉ số thông minh cao. Nhưng đôi khi hay nghi ngờ."
    },
    {
      "name":"Big",
      "label": "Mắt to",
      "analysis":"Tốt bụng và giàu lòng trắc ẩn, có thể là bạn tốt để trò chuyện. Hiểu biết tốt về điện ảnh và nghệ thuật, đôi khi bị cảm xúc lấn át."
    }
    ]
  },
  {
    "name":"face",
    "features":[
    {
      "name":"Triangle",
      "label": "Khuôn mặt tam giác",
      "analysis":"Nghiêm khắc với người khác cũng như với chính mình, làm việc gì cũng cẩn thận. Họ dường như không thân thiết với mọi người."
    },
    {
      "name":"Oval",
      "label": "Khuôn mặt oval",
      "analysis":"Có nhiều khả năng cải thiện tốt hơn sự nghiệp của họ. Họ cũng tiêu tiền nhiều hơn và quan tâm đến chất lượng cuộc sống."
    },
    {
      "name":"Square",
      "label": "Khuôn mặt vuông",
      "analysis":"Hoạt bát, cương quyết, dứt khoát và hết sức quan tâm đến người mình thích, nhưng hay gặp phải biến cố trong cuộc đời."
    },
    {
      "name":"Circle",
      "label": "Khuôn mặt tròn",
      "analysis":"Thường khá vui tính, luôn vui vẻ, sẵn lòng giúp đỡ người khác, sống rất thân thiện hòa nhập với mọi người vì thế mà được mọi người vô cùng yêu mến. Tuy nhiên đôi lúc họ lại tùy tiện, cá nhân."
    }
    ]
  },
  {
    "name":"mouth",
    "features":[
    {
      "name":"Small",
      "label": "Miệng nhỏ",
      "analysis":"Tính cách hướng nội và bảo thủ. Họ thường thua thiệt khi gặp khó khăn."
    },
    {
      "name":"Medium",
      "label": "Miệng trung bình",
      "analysis":"Có mối quan hệ tốt với những người khác và tốt bụng với mọi người. May mắn trong cả sự nghiệp và tài lộc."
    },
    {
      "name":"Thick",
      "label": "Miệng dày",
      "analysis":"Sinh ra để chăm nom người khác. Hiền lành tốt bụng, người yêu động vật, sẵn lòng giúp đỡ người khác khi họ gặp khó khăn. Thường nghĩ cho người khác rồi mới nghĩ đến bản thân."
    }
    ]
  },
  {
    "name":"nose",
    "features":[
    {
      "name":"Wide",
      "label": "Mũi rộng",
      "analysis":"Quyết đoán, năng động, có dũng khí và tài năng, với một chút tâm lý đầu cơ. Có đầu óc tốt và sẵn sàng làm việc chăm chỉ, nhưng có lòng tự trọng cao và quyền lực đáng kể, tình bạn rộng rãi và thích thể diện, không tiếc công sức theo đuổi sự giàu có."
    },
    {
      "name":"Long",
      "label": "Mũi dài",
      "analysis":"Có khả năng tư duy sâu sắc, nhưng lại hay suy nghĩ, đắn đo. Những người này không thích giãi bày tâm sự. Họ chú ý đến việc bồi bổ tinh thần nhiều hơn. Họ có nhiều sở thích nhưng khá bảo thủ. Họ thường sống ổn định và cô đơn."
    },
    {
      "name":"Small",
      "label": "Mũi nhỏ",
      "analysis":"Người có mũi nhỏ là người chu đáo, nhạy cảm, tính tình bảo thủ, ôn hòa, ngại hành động, ít tham vọng, đến tuổi trung niên tương đối bết bát. Đàn ông phải làm việc chăm chỉ hơn trong công việc, còn phụ nữ thì hơi chậm trễ trong hôn nhân."
    }
    ]
  }
  ]
}
```

<div>
<div class="screen-tv">
<a class="image-link" href="https://pwieu.com/click-FQLMKJP1-KHEQCJKZ?bt=25&tl=1&url=https%3A%2F%2Fshopee.vn%2Fp-i.371881430.5280975247"><img src="/assets/img/ads/4m-solar-robot.gif"></a>
</div>
<img class="cabinet-img" src="/assets/img/cabinet-tv.png">
</div>


### Thư mục models

Tải pretrained model về thư mục models

```
cd models
wget https://github.com/YeyunLU/Face2Fate/raw/master/src/CNN/models/eye_model.pt
wget https://github.com/YeyunLU/Face2Fate/raw/master/src/CNN/models/eyebrow_model.pt
wget https://github.com/YeyunLU/Face2Fate/raw/master/src/CNN/models/jaw_model.pt
wget https://github.com/YeyunLU/Face2Fate/raw/master/src/CNN/models/mouth_model.pt
wget https://github.com/YeyunLU/Face2Fate/raw/master/src/CNN/models/nose_model.pt
pip install gdown
gdown gdown https://drive.google.com/uc?id=1-JGQ1B9w6dteDHJPNwp-YWDGMcPO16LL
```

### Chạy chương chình
```
python main.py
```

<iframe width="560" height="315" src="https://www.youtube.com/embed/Z1vqlGMZrmg" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


### Nguồn tham khảo

[facial-beauty-prediction](https://github.com/etrain-xyz/facial-beauty-prediction)

[Face2Fate](https://github.com/YeyunLU/Face2Fate)