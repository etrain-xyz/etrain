---
title: "Face reading"
date: 2021-02-16 23:16:23
image: '/assets/img/posts/face-reading.png'
description: 'Face reading using Deep learning'
main-class: 'funny'
tags:
- python
- deep learning
- face recognition
categories:
- 'Face reading'
twitter_text: 'Face reading using Deep learning'
introduction: 'Face reading using Deep learning'
---

In [this post](/en/posts/facial-beauty-prediction) Etrain introduced the face beauty score model. Today, Etrain introduce the [face2fate](https://github.com/YeyunLU/Face2Fate) repo, we can predict a person's personality via their facial features.

![Face Reading](/assets/img/posts/face-reading.png)


Project structure:

![Face Reading Project Structure](/assets/img/posts/face-reading-project-sctructure.png)


### The data directory

Download `shape_predictor_5_face_landmarks.dat.bz2` and `shape_predictor_68_face_landmarks.dat.bz2` into the `data` directory

```
cd data
wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
bzip2 -d shape_predictor_5_face_landmarks.dat.bz2
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

Create `analysis.json` file in the `data` directory

```json
{
  "face_regions":
  [
    {
      "name":"eyebrows",
      "features":[
        {
          "name":"Straight",
          "label":"Straight",
          "analysis":"Usually with great perseverance and courage. People with thick straight eyebrows are often strong-willed, courageous and stern. People with thin straight eyebrows are often clever, talented and shrewd."


        },
        {
          "name":"Arch",
          "label":"Arch",
          "analysis":"Energetic, not afraid of difficulties, brave and confident to innovate. But sometimes may be stubborn."
        },
        {
          "name":"Circle",
          "label":"Circle",
          "analysis":"Being mild, friendly and considerate, usually with good relationship with others. Have stronger feelings for art, but sometimes will be emotional and idealistic in real life."
        }
      ]
    },
    {
      "name":"eyes",
      "features":[
        {
          "name":"Small",
          "label":"Small",
          "analysis":"Be more rational, are more suitable for science research. Have their own ideas, not easily affected by others."
        },
        {
          "name":"Slit",
          "label":"Slit",
          "analysis":"Often with good appearance higher intelligence quotient. But sometimes may be suspicious to other things."
        },
        {
          "name":"Big",
          "label":"Big",
          "analysis":"Kind and compassionate, can be good friends to talk with. Good understanding of movie and arts, sometime overwhelmed by emotions."
        }
      ]
    },
    {
      "name":"face",
      "features":[
        {
          "name":"Triangle",
          "label":"Triangle",
          "analysis":"Be strict with others as well as themselves, do things carefully. They may seems to be not close to others. "
        },
        {
          "name":"Oval",
          "label":"Oval",
          "analysis":"More likely to have better improvement in their careers. They also spend more money and pay attention to quality of life."
        },
        {
          "name":"Square",
          "label":"Square",
          "analysis":"Energetic, resolute, decisive and take great care of the people they like, but sometimes blindly advance."
        },
        {
          "name":"Circle",
          "label":"Circle",
          "analysis":"Kind, friendly and sympathetic. Usually have good relationship with friends. More likely to be attractive by other."
        }
      ]
    },
    {
      "name":"mouth",
      "features":[
        {
          "name":"Small",
          "label":"Small",
          "analysis":"A more introverted and conservative personality. They are often at a loss when encountering difficulties."
        },
        {
          "name":"Medium",
          "label":"Medium",
          "analysis":"Have good relationship with others and are kind of popular. Being luck for both career and wealth."
        },
        {
          "name":"Thick",
          "label":"Thick",
          "analysis":"Passionate and warm-hearted, emphasizing sensory stimuli, doing things practically and resentful, and less likely to use tricks."
        }
      ]
    },
    {
      "name":"nose",
      "features":[
        {
          "name":"Wide",
          "label":"Wide",
          "analysis":"Decisive, energetic, and have the courage and talent, with a little speculative mentality. Good-minded and willing to work hard, but with strong self-esteem and considerable power, broad friendships and love for face, spare no effort in the pursuit of wealth."
        },
        {
          "name":"Long",
          "label":"Long",
          "analysis":"People with long noses have deep thinking skills, but often think more and hesitate. I don't like to confide in my mind. I pay more attention to spiritual enrichment. I have a wide range of interests but are conservative. I am stable and lonely."
        },
        {
          "name":"Small",
          "label":"Small",
          "analysis":"People with small noses are thoughtful and sensitive, have a conservative and peaceful personality, hesitate to act, have less ambition, and are relatively flat in middle age. Men must work harder at work, and women are slightly delayed in marriage some."
        }
      ]
    }
  ]
}
```

### The models directory

Download pretrained model into the `models` directory

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

### Running
```
python main.py
```

<iframe width="560" height="315" src="https://www.youtube.com/embed/Z1vqlGMZrmg" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


### References

[facial-beauty-prediction](https://github.com/etrain-xyz/facial-beauty-prediction)

[Face2Fate](https://github.com/YeyunLU/Face2Fate)