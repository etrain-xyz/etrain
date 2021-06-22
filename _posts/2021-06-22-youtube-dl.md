---
title: "Tải video youtube với công cụ youtube-dl"
date: 2021-06-22 15:22:00
image: '/assets/img/posts/youtube-dl.jpg'
description: 'Tải video youtube với công cụ youtube-dl'
main-class: 'guide'
tags:
- python
- youtube downloader
categories:
twitter_text: 'Tải video youtube với công cụ youtube-dl'
introduction: 'Tải video youtube với công cụ youtube-dl'
---

Bạn có thể dễ dàng tải video trên YouTube với công cụ youtube-dl. Với công cụ này, bạn cũng có thể chọn định dạng video và chất lượng video muốn tải về như 1080p hoặc 4K.

```bash
## Cài đặt
pip install youtube_dl

## Liệt kê danh sách định dạng và chất lượng video
## youtube-dl -F <video_url>
youtube-dl -F https://www.youtube.com/watch?v=9LS4XCNyRKA

## Output
# [youtube] 9LS4XCNyRKA: Downloading webpage
# [info] Available formats for 9LS4XCNyRKA:
# format code  extension  resolution note
# 249          webm       audio only tiny   51k , webm_dash container, opus @ 51k (48000Hz), 926.50KiB
# 250          webm       audio only tiny   57k , webm_dash container, opus @ 57k (48000Hz), 1.03MiB
# 251          webm       audio only tiny  110k , webm_dash container, opus @110k (48000Hz), 1.95MiB
# 140          m4a        audio only tiny  129k , m4a_dash container, mp4a.40.2@129k (44100Hz), 2.29MiB
# 160          mp4        256x144    144p   90k , mp4_dash container, avc1.4d400c@  90k, 25fps, video only, 1.61MiB
# 278          webm       256x144    144p   96k , webm_dash container, vp9@  96k, 25fps, video only, 1.71MiB
# 242          webm       426x240    240p  180k , webm_dash container, vp9@ 180k, 25fps, video only, 3.20MiB
# 133          mp4        426x240    240p  228k , mp4_dash container, avc1.4d4015@ 228k, 25fps, video only, 4.04MiB
# 243          webm       640x360    360p  307k , webm_dash container, vp9@ 307k, 25fps, video only, 5.45MiB
# 134          mp4        640x360    360p  443k , mp4_dash container, avc1.4d401e@ 443k, 25fps, video only, 7.86MiB
# 244          webm       854x480    480p  514k , webm_dash container, vp9@ 514k, 25fps, video only, 9.12MiB
# 135          mp4        854x480    480p  791k , mp4_dash container, avc1.4d401e@ 791k, 25fps, video only, 14.02MiB
# 247          webm       1280x720   720p  985k , webm_dash container, vp9@ 985k, 25fps, video only, 17.46MiB
# 136          mp4        1280x720   720p 1459k , mp4_dash container, avc1.64001f@1459k, 25fps, video only, 25.86MiB
# 18           mp4        640x360    360p  537k , avc1.42001E, 25fps, mp4a.40.2 (44100Hz), 9.53MiB
# 22           mp4        1280x720   720p 1588k , avc1.64001F, 25fps, mp4a.40.2 (44100Hz) (best)

## Tải về video có format code là 136
## (định dạng MP4, độ phân giải 1280x720)
youtube-dl -f 136 https://www.youtube.com/watch?v=9LS4XCNyRKA
```

<div>
<div class="screen-tv">
<a class="image-link" href="https://pwieu.com/click-FQLMKJP1-KHEQCJKZ?bt=25&tl=1&url=https%3A%2F%2Fshopee.vn%2Fp-i.299252.2562488186"><img src="/assets/img/ads/Mijia-Bicycle-Pump-MJCQB01QJ.gif"></a>
</div>
<img class="cabinet-img" src="/assets/img/cabinet-tv.png">
</div>


Xem thêm về cách sử dụng youtube-dl [tại đây](https://github.com/ytdl-org/youtube-dl/blob/master/README.md#readme)