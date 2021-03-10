---
title: "Download kaggle data in Google Colab"
date: 2021-03-10 10:01:00
image: '/assets/img/posts/kaggle-data.png'
description: 'Download kaggle data in Google Colab'
main-class: 'guide'
tags:
- python
- kaggle dataset
- google colab
categories:
twitter_text: 'Download kaggle data in Google Colab'
introduction: 'Download kaggle data in Google Colab'
---

Please follow the steps below to download and use kaggle data within Google Colab

### Install kaggle on Google Colab

- Login and go to [your account](https://www.kaggle.com/me/account).
- Scroll to API section and Click **Create New API Token**. It will download `kaggle.json` file on your machine.

![Kaggle Account API](/assets/img/posts/kaggle-account-api.png)

- Go to your Google Colab project file and run the following commands

```
!pip install -q kaggle

from google.colab import files
files.upload()
# Choose the kaggle.json file that you downloaded

!mkdir '/root/.kaggle'
!cp kaggle.json '/root/.kaggle'
!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets list
# That's all ! You can check if everything's okay by running this command.
```

### Download Data

Now you can download any dataset in kaggle. For example we will download [VinBigData Original Image Dataset](https://www.kaggle.com/awsaf49/vinbigdata-original-image-dataset).

Click to the menu and click **Copy API command**

![Kaggle Dataset API Command](/assets/img/posts/kaggle-data-api-cmnd.png)


Paste the command to Google Colab and run.

```
!kaggle datasets download -d awsaf49/vinbigdata-original-image-dataset
```

Unzip and using it

```
!unzip -q vinbigdata-original-image-dataset.zip
```

### References
[Easiest way to download kaggle data in Google Colab](https://www.kaggle.com/general/74235)