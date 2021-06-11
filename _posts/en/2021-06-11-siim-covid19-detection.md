---
title: "COVID-19 abnormalities on chest radiographs"
date: 2021-06-11 15:22:00
image: '/assets/img/posts/siim-covid19.png'
description: 'COVID-19 detection EfficientNetV2'
main-class: 'competition'
tags:
- python
- deep learning
- EfficientNetV2
categories:
twitter_text: 'COVID-19 detection EfficientNetV2'
introduction: 'COVID-19 detection EfficientNetV2'
---

### Prepare data

[SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/c/siim-covid19-detection/overview/description) is the competition that you’ll identify and localize COVID-19 abnormalities on chest radiographs. In particular, you'll categorize the radiographs as negative for pneumonia or typical, indeterminate, or atypical for COVID-19. You and your model will work with imaging data and annotations from a group of radiologists.

The images are in DICOM format, Etrain convert image to JPG, you can download data in [this notebook](https://www.kaggle.com/duythanhng/siim-covid-19-convert-to-jpg-original)

In the Input section you download `train_image_level.csv` and `train_study_level.csv`. In the Output section you download `meta.csv` and `train.tar.gz`.

```
siim-covid
|-- train_image_level.csv
|-- train_study_level.csv
|-- train.tar.gz
|-- meta.csv
```

Extract train.tar.gz

```bash
!tar xzf file.tar.gz
```

Create train and valid dataset

```python
import pandas as pd

train_image_level = pd.read_csv("/content/drive/MyDrive/Etrain/siim-covid/train_image_level.csv")
train_study = pd.read_csv('/content/drive/MyDrive/Etrain/siim-covid/train_study_level.csv')

# merge study csv
train_study['StudyInstanceUID'] = train_study['id'].apply(lambda x: x.replace('_study', ''))
del train_study['id']
train = train_image_level.merge(train_study, on='StudyInstanceUID')

train = train.rename(columns={
    'Negative for Pneumonia': 'negative',
    'Typical Appearance': 'typical',
    'Indeterminate Appearance': 'indeterminate',
    'Atypical Appearance': 'atypical'
})

def get_image_name(image_id):
    image_name = image_id.split("_image")[0] + ".jpg"
    return image_name

train["image"] = train["id"].apply(get_image_name)
```

### EfficientNetV2

We use EffNetV2-L and `efficientnetv2-l-21k-ft1k` pretrained

![EfficientNetV2](/assets/img/posts/efficientnetv2.png)

```python
import tensorflow as tf
import tensorflow_hub as hub

print('TF version:', tf.__version__)
print('Hub version:', hub.__version__)
print('Phsical devices:', tf.config.list_physical_devices())

# Build model
hub_url = 'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-l-21k-ft1k/feature-vector'
image_size = 480
batch_size = 4
labels = ["negative", "typical", "indeterminate", "atypical"]

tf.keras.backend.clear_session()
model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=[image_size, image_size, 3]),
    hub.KerasLayer(hub_url, trainable=True),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(
        len(labels),
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        activation='sigmoid'
    )
])
model.build((None, image_size, image_size, 3))
model.summary()
```

![Model Summary](/assets/img/posts/siim-covid19-model-summary.png)

We use `ImageDataGenerator` to create the train and valid dataset. With the train dataset we use the data augmentation technique (horizontal_flip and vertical_flip).

```python
datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=(image_size, image_size),
                       batch_size=batch_size)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      horizontal_flip=True,
      vertical_flip=True,
      **datagen_kwargs)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)

image_path = '/content/drive/MyDrive/Etrain/siim-covid/train'

train_generator = train_datagen.flow_from_dataframe(
    train,
    directory=image_path,
    shuffle=True,
    class_mode="raw",
    color_mode="rgb",
    x_col="image",
    y_col=labels,
    subset="training",
    **dataflow_kwargs
)

valid_generator = valid_datagen.flow_from_dataframe(
    train,
    directory=image_path,
    shuffle=False,
    class_mode="raw",
    color_mode="rgb",
    x_col="image",
    y_col=labels,
    subset="validation",
    **dataflow_kwargs
)
```

Train model and save the checkpoint with the minimum val_loss value

```python
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="/content/drive/MyDrive/Etrain/siim-covid/efficientnetv2-l-21k-ft1k-study-level.h5", 
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
    mode='min'
)

model.compile(
	optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
	loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1),
	metrics=['accuracy']
)

hist = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=valid_generator,
    callbacks=[model_checkpoint]).history
```

![Training result](/assets/img/posts/siim-covid19-training.png)


You can see [this notebook](https://www.kaggle.com/duythanhng/siim-covid-19-efficientnetv2-infer) to submit the prediction

### References

[EfficientNetV2](https://github.com/google/automl/tree/master/efficientnetv2)

