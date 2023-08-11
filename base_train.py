# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from PIL import ImageFont, ImageDraw, Image
from fontTools.ttLib import TTFont

import numpy as np
import pandas as pd
from models import *

# -

# ## Load Data

class Glyph(object):
    # transform character to bitmap
    def __init__(self, fonts, size=64):
        # load fonts, size. We will use 2 fonts for all CJK characters, so keep 2 codepoint books.
        self.codepoints = [set() for _ in fonts]
        self.size = int(size * 0.9)
        self.size_img = size
        self.pad = (size - self.size) // 2
        self.fonts = [ImageFont.truetype(f, self.size) for f in fonts]
        # use a cache to reduce computation if duplicated characters encountered.
        self.cache = {}
        for cp, font in zip(self.codepoints, fonts):
            font = TTFont(font)
            # store codepoints in font cmap into self.codepoints
            for cmap in font['cmap'].tables:
                if not cmap.isUnicode():
                    continue
                for k in cmap.cmap:
                    cp.add(k)
    
    def draw(self, ch):
        if ch in self.cache:
            return self.cache[ch]
        # search among fonts, use the first found
        exist = False
        for i in range(len(self.codepoints)):
            if ord(ch) in self.codepoints[i]:
                font = self.fonts[i]
                exist = True
                break
        if not exist:
            return None

        img = Image.new('L', (self.size_img, self.size_img), 0)
        draw = ImageDraw.Draw(img)
        (width, baseline), (offset_x, offset_y) = font.font.getsize(ch)
        draw.text((self.pad - offset_x, self.pad - offset_y + 4), ch, font=font, fill=255, stroke_fill=255) 
        img_array = np.array(img.getdata(), dtype='uint8').reshape((self.size_img, self.size_img))
        self.cache[ch] = img_array

        return img_array


def tokenizer(code):
    # Cangjie code consists only of a-z, and seperator ','
    # start, end will be designed with special tokens.
    tokens = [1]
    for c in code:
        if c >= 'a' and c <= 'z':
            tokens.append(ord(c) - 93)
        elif c == ',':
            tokens.append(3)
        else:
            print(f'Invalid code: {c} found in {code}.')
            return None
    tokens.append(2)
    return np.array(tokens, dtype='uint8')


def produce_data(img_size=128):
    try:
        glyphs, tokens = pickle.load('data.pt')
        print('Loaded from data.pt')
    
    except:
        glyphbook = Glyph(['data/fonts/TH-Tshyn-P0.ttf', 'data/fonts/TH-Tshyn-P1.ttf', 'data/fonts/TH-Tshyn-P2.ttf'],
                      size=img_size)
        code_chart = pd.read_csv('data/Cangjie5.txt', delimiter='\t', header=None, names=['Char', 'Code', 'Note'], keep_default_na=False)
        transformed_chart = code_chart.groupby('Char').apply(lambda x: ','.join(x['Code'].to_list()))
        transformed_chart = transformed_chart.sample(frac=1, random_state=2023)

        glyphs = []
        tokens_src = []
        tokens_tgt = []
        for char, code in transformed_chart.reset_index().to_numpy():
            glyph = glyphbook.draw(char)
            token = tokenizer(code)
            if glyph is not None and token is not None:
                glyphs.append(glyph)
                tokens_src.append(token[:-1])
                tokens_tgt.append(token[1:])
            else:
                print(f'WARNING: {char}: {code} cannot be processed')

    return glyphs, tokens_src, tokens_tgt

def create_dataset(img_size):
    glyphs, tokens_src, tokens_tgt = produce_data(img_size=img_size)
    data_x = tf.data.experimental.from_list(list(zip(glyphs, tokens_src)))
    data_y = tf.data.experimental.from_list(tokens_tgt)
    data = tf.data.Dataset.zip((data_x, data_y))
    return data

IMG_SIZE = 128
MAX_LEN = 64
VOCAB_SIZE = 30
BATCH_SIZE = 128

try:
    data = tf.data.Dataset.load('compiled_data', compression='GZIP')
except:
    data = create_dataset(IMG_SIZE)
    data.save('compiled_data', compression='GZIP')

train_size = int(0.9 * len(data))
train_data = data.take(train_size)
test_data = data.skip(train_size)


def preprocess(dataset, batch_size):
    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


train_data = preprocess(train_data, batch_size=BATCH_SIZE)
test_data = preprocess(test_data, batch_size=BATCH_SIZE)


# ## Model

class SparseCategoricalCrossentropy(keras.losses.Loss):
    
    def __init__(self, ignore_class=None, label_smoothing=0.0, name='loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ignore_class = ignore_class
        self.label_smoothing = label_smoothing
        self.kwargs = kwargs
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true_onehot = tf.one_hot(y_true, tf.shape(y_pred)[-1], dtype=y_pred.dtype)
        losses = keras.losses.categorical_crossentropy(y_true_onehot, y_pred, label_smoothing=self.label_smoothing, **self.kwargs)
        if self.ignore_class is not None:
            losses *= tf.cast(y_true != self.ignore_class, dtype=y_pred.dtype)
        if sample_weight is not None:
            losses *= tf.cast(sample_weight, y_pred.dtype)
        
        return tf.reduce_mean(tf.reduce_sum(losses, axis=-1), axis=0)

    def get_config(self):
        config = {"ignore_class": self.ignore_class, "label_smoothing": self.label_smoothing, **self.kwargs}
        return config
    
    @classmethod
    def from_config(self, config):
        return SparseCategoricalCrossentropy(**config)


class SparseCategoricalAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='code_wise_accuracy', ignore_class=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = self.add_weight(name='count', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
        self.ignore_class = ignore_class

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.ignore_class is not None:
            self.total.assign_add(tf.reduce_sum(tf.cast(y_true != self.ignore_class, self.dtype)))
        else:
            self.total.assign_add(tf.cast(tf.reduce_prod(tf.shape(y_true)), self.dtype))
            
        y_pred_labels = tf.cast(tf.argmax(y_pred, axis=-1), y_true.dtype)
        values = tf.logical_and(y_true == y_pred_labels, y_true != self.ignore_class)
        self.count.assign_add(tf.reduce_sum(tf.cast(values, self.dtype)))

    def result(self):
        return self.count / self.total

    def reset_state(self):
        self.count.assign(0)
        self.total.assign(0)


# ## Graph

model = Cangjie(VOCAB_SIZE, MAX_LEN)

# +
EPOCHS = 100

model.compile(keras.optimizers.Adam(learning_rate=0.00005), loss=SparseCategoricalCrossentropy(ignore_class=0, label_smoothing=0.01),
              metrics=[SparseCategoricalAccuracy(ignore_class=0)], jit_compile=False)

model.summary()

# ## Training

log_dir = './logs/cangjie'

model.fit(train_data, epochs=EPOCHS, validation_data=test_data, callbacks=[
    keras.callbacks.TensorBoard(log_dir=log_dir)])

model.save('cangjie', save_format='tf')
