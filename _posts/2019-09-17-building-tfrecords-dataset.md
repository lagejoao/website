---
toc: true
layout: post
description: "How do I changed my dataset creation method"
categories: [tensorflow, keras, ml]
title: "Building a TFRecords dataset"
permalink: /building-tfrecords-dataset/
author: Joao Lage
---

# Introduction

> For a long time I worked with pickle, JSON and CSV files in order to build a dataset that could be used to train a ML model.

We all know the importance to have reliable, reusable and portable datasets when training any model.

As we know, pickle files show some disadvantages:

* No compression
* Assumes you will have the same packages when decoding the file (example: numpy)
* You cannot stream it from disk since you have to read and decode the file/part as a whole first

**What happens next?** - Let me present you TFRecords!

# Creating the Dataset

What is a TFRecords file?

> "To read data efficiently it can be helpful to serialize your data and store it in a set of files (...)"
> src: [TFRecords tutorials](https://www.tensorflow.org/tutorials/load_data/tf_records)

TFRecords take advantage of [protocol buffers](https://developers.google.com/protocol-buffers/) to efficiently create a cross-platform structure of the data to be saved. The saved data is a sequence of binary records.


### How do I start?
Let's pretend our dataset is made of images, descriptions and labels. Labels will be our targets for future use.
First we will need to define a protobuf and `tf.train.Example` will ease that work for us:


```python
import tensorflow as tf


def serialize_example(image, description, label):
    feature = {
        "description_index": tf.train.Feature(
            int64_list=tf.train.Int64List(value=description.indices)
        ),
        "description_value": tf.train.Feature(
            int64_list=tf.train.Int64List(value=description.data)
        ),
        "image": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image])
        ),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
```

The above snippet creates a structure holding a sparse vector representation for textual descriptions, a byte string for images and a list for labels . If you want to encode your text as a sequence you can easily do it without the need of having both indices and values - just values will be enough.

In regard to images, this example assumes that each image was downloaded, resized and saved previously into disk and can be loaded through its image path.

At the moment, TFRecords support [three generic types](https://www.tensorflow.org/tutorials/load_data/tf_records#data_types_for_tfexample), `BytesList`, `FloatList` and `Int64List`, but they can be coerced into many other data types.

## How do I create a TFRecords file?

After having the dataset split and the vectorisers fitted, you can loop through the data splits, create a binary representation of each data point and save it to a `.tfrecords` file:

```python
import os
import cv2


datasets = [(train, "train"), (val, "val"), (test, "test")]
for dataset, file_name in datasets:
    file_path = os.path.join(output_dir, file_name + ".tfrecords")
    with tf.python_io.TFRecordWriter(file_path) as writer:
        for row in dataset:
            img = cv2.imread(row["image_path"]).tobytes()

            description = text_encoder.transform(
                [row["description"]]
            )

            label_encoded = label_encoder.transform(row["label"])

            example = serialize_example(image, description, label)
            writer.write(example)
```

For reproducibility purposes, `text_encoder` is a [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) object and `label_encoder` is a [LabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html) object.

{% include info.html text="In case of using encoders for text and/or targets, I recommend to save their configuration in order to dynamically load the number of unique tokens and/or classes when training a model." %}

After running the snippets above you will create three files `train.tfrecords`, `val.tfrecords` and `test.tfrecords`!


## How do I read a TFRecord file?
First we need to create a representation of the feature we want to decode from each TFRecord:

```python
feature_description = {
    'label': tf.FixedLenFeature([], tf.int64, default_value=0),
    'image': tf.FixedLenFeature([], tf.string),
    'description': tf.SparseFeature(index_key="description_index", value_key="description_value", dtype=tf.int64, size=description_max_features),
}
```

Luckily we have the ability to turn an array of indices and an array of values into a sparse feature without much hassle with `tf.SparseFeature`, just by specifying where to read the indices, values and the maximum length of the sparse representation. Please note that for every example: `len(indices) == len(values)` and `indices` should be contained in `[0,...,max_features-1]`.

Then we create a parsing function that reads the byte string of each example and decodes it:

```python
def _parse_function(example_proto):
    example = tf.parse_single_example(example_proto, feature_description)

    label = example["label"]
    description = example["description"]
    
    image_shape = tf.constant([128, 128, 3])
    image = tf.decode_raw(example["image"], tf.uint8)
    image = tf.reshape(image, image_shape)
    
    return {
        "description": description,
        "image":image,
    }, label
```

**Tip:** If you wish to turn a sparse tensor into dense you can use [tf.sparse.to_dense](https://www.tensorflow.org/api_docs/python/tf/sparse/to_dense).

**Tip:** Using pre-trained imagenet weights?

If using an image model initialised with imagenet weights you should subtract the channel mean on each pixel using the below function. Also, channels will be swapped from RGB to BGR. This preprocessing stage guarantees that you will be sourcing images to the model following the same pixel value distribution per channel from the imagenet's dataset.

The code block below is a simplified version of the current Keras [implementation](https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L93) that assumes the `x` tensor to have color channels in the last dimension and channels input order should be RGB.

```python
def preprocess_symbolic_input(x):
    backend = tf.keras.backend
    data_format = "channels_last"

    # 'RGB'->'BGR'
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]

    mean_tensor = backend.constant(-np.array(mean))

    # Zero-center by mean pixel
    if backend.dtype(x) != backend.dtype(mean_tensor):
        x = backend.bias_add(
            backend.cast(x, backend.dtype(mean_tensor)),
            mean_tensor,
            data_format=data_format,
        )
    else:
        x = backend.bias_add(x, mean_tensor, data_format="channels_last")

    return x
```

Finally, we create an helper to reads the datasets:
    
```python
def read_dataset(file, shuffle=False, batch_size=64, prefetch=5):
    dataset = tf.data.TFRecordDataset(file)
    if shuffle:
        dataset = dataset.shuffle()
    return dataset.map(_parse_function)

train_dataset = read_dataset("train.tfrecords")
val_dataset = read_dataset("val.tfrecords")
test_dataset = read_dataset("test.tfrecords")
```


# Fitting a model
A TFRecord dataset is also know by its versatility since it can be used when fitting a `tf.keras.Model` (see [docs](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)) and also when training an instance of `tf.estimator.Estimator` (see: [docs](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator) and [guide](https://www.tensorflow.org/guide/estimators)).

```python
def input_fn_train(model_path, batch_size):
    dataset = read_dataset(model_path)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=5)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(buffer_size=5)


def input_fn_eval(model_path, batch_size):
    dataset = read_dataset(model_path)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    return dataset.prefetch(buffer_size=5)
```

Voilà! You are now able to feed batches of tfrecords into your model!