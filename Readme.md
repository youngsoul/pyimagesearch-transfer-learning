# PyImageSearch Transfer Learning with Keras and Deep Learning

Working through the blog below and doing my own investigation.

https://www.pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/

Create a 'Food', 'NOT-Food' image classifier using Transfer Learning


## DataSet
The original blog post pointed to

https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/

for the dataset, but I had trouble downloading from there.

I ended up using the Kaggle Food-5k which is close enough for the purpose of this exercise.

https://www.kaggle.com/binhminhs10/food5k

## My Modifications

The original blog post was setup to only use VGG16 and I was curious how other CNNs might perform.  

I created the file `cnn_models.py` which has an array of models to try.

```python
from tensorflow.keras.applications import VGG16, ResNet50V2
from tensorflow.keras.layers import Input

MODELS = [
    {
        "base_model": VGG16(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(224, 224, 3))),
        "name": "vgg16",
        "feature_shape": 7 * 7 * 512
    },
    {
        "base_model": ResNet50V2(weights="imagenet", include_top=False,
                                 input_tensor=Input(shape=(224, 224, 3))),
        "name": "resnet50v2",
        "feature_shape": 7 * 7 * 2048

    }
]

```

You can add multiple models here and the `ml_train.py` script will run through all of these models.

the `config.py` was changed slightly to take the model name as a path parameter so all of the feature CSV files are saved in a directory with the model name.

## Validation Prediction

I added the file, `ml_predict.py` which starts with the raw validation images and goes through the process of creating the feature vector and then uses the saved LogisticRegression model from the `ml_train.py` script to make prediction.

The `ml_predict.py` will then display the validation images that were predicted incorrectly.

Of the 1000 Validation images here are the resultsl

```text
Accuracy: 0.99
Total Images: 1000
Total Errors: 14

```

Here is sample of just the errors.  

![errors](./doc_images/2020-03-20_19-03-17%20(1).gif)