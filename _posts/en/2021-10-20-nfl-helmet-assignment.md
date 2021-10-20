---
title: "NFL Helmet Assignment"
date: 2021-07-03 08:00:00
image: '/assets/img/posts/nfl-helmet-assignment.jpg'
description: 'NFL Helmet Assignment'
main-class: 'guide'
tags:
- python
- deep learning
- object detection
- multi-object tracking
categories:
twitter_text: 'NFL Helmet Assignment'
introduction: 'NFL Helmet Assignment'
---

### Introduction
[The 2021 NFL Health & Safety, Helmet Assignment competition](https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment) is part of a collaborative effort between National Football League (NFL) and Amazon Web Services (AWS) to assist in the development of the best sports injury surveillance and mitigation program in the world. We are trying to assign the correct player "label" on helmets in NFL game footage. Labels consist of a value H (for home team) and V (for visiting team) followed by the player's jersey number. Player labels are provided within the Next Gen Stats (NGS) tracking data for each play. A perfect submission would correctly identify the helmet box for every helmet in every frame of video and assign that helmet the correct player label.


### Solution

The approach is fairly simple:
1. [Step through each frame in a video and apply the deepsort algorithm](https://www.kaggle.com/duythanhng/nfl-helmet-with-yolov5-deepsort-starter). This clusters helmets across frames when it is the same player/helmet.
2. [Group by each of these deepsort clusters - and pick the most common label for that cluster](https://www.kaggle.com/duythanhng/nfl-yolov5-deepsort-pytorch-guide). Then override all of the predictions for that helmet to the same player.

<iframe width="560" height="315" src="https://www.youtube.com/embed/TofMADTFkjI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### References
[NFL Helmet Assignment - Getting Started Guide](https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide)

[Helper Code + Helmet Mapping + Deepsort](https://www.kaggle.com/robikscube/helper-code-helmet-mapping-deepsort)

[NFL helmet with Yolov5-deepsort starter](https://www.kaggle.com/s903124/nfl-helmet-with-yolov5-deepsort-starter)