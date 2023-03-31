import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot

class PosEnc2D(layers.Layer, tfmot.sparsity.keras.PrunableLayer):

    def __init__(self, embed_dim=512, n=10000, **kwargs):
        super(PosEnc2D, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.n = n
        
    def build(self, input_shape):
        incremental = ((self.embed_dim + 3) // 4) * 2
        self.base_fq = 1 / (tf.cast(self.n, self.dtype) ** (tf.range(0, incremental, 2, dtype=self.dtype) / incremental))

    def cal_enc(self, shape):
        half_channels = tf.shape(self.base_fq)[0]
        x_position = tf.einsum('x,d->xd', tf.range(0, shape[1], dtype=self.dtype), self.base_fq)
        y_position = tf.einsum('x,d->xd', tf.range(0, shape[2], dtype=self.dtype), self.base_fq)
        x_encoding = tf.reshape(tf.stack([tf.sin(x_position), tf.cos(x_position)], axis=-1), [shape[1],  2 * half_channels])
        y_encoding = tf.reshape(tf.stack([tf.sin(y_position), tf.cos(y_position)], axis=-1), [shape[2],  2 * half_channels])
        encoding = tf.concat([tf.tile(x_encoding[:,tf.newaxis,:], (1,shape[2],1)),
                              tf.tile(y_encoding[tf.newaxis,:,:], (shape[1],1,1))], axis=-1)
        return encoding[:,:self.embed_dim]

    def call(self, x):
        shape = tf.shape(x)
        return self.cal_enc(shape)
    
    def get_prunable_weights(self):
        return []
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "n": self.n})
        return config

class PosEnc1D(layers.Layer, tfmot.sparsity.keras.PrunableLayer):

    def __init__(self, embed_dim=512, n=10000, **kwargs):
        super(PosEnc1D, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.n = n
        self.shape = 0
        
    def build(self, input_shape):
        incremental = ((self.embed_dim + 1) // 2) * 2
        self.base_fq = 1 / (tf.cast(self.n, self.dtype) ** (tf.range(0, incremental, 2, dtype=self.dtype) / incremental))

    def cal_enc(self, shape):
        position = tf.einsum('x,d->xd', tf.range(0, shape[1], dtype=self.dtype), self.base_fq)
        encoding = tf.reshape(tf.stack([tf.sin(position), tf.cos(position)], axis=-1), [shape[1], 2 * tf.shape(self.base_fq)[0]])
        return encoding[:,:self.embed_dim]
    
    def call(self, x):
        shape = tf.shape(x)
        return self.cal_enc(shape)
    
    def get_prunable_weights(self):
        return []
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "n": self.n})
        return config


class ImgEmbed(layers.Layer, tfmot.sparsity.keras.PrunableLayer):

    def __init__(self, embed_dim=512, ff_dim=2048, glyph_block_size=16, rate=0.1, **kwargs):
        super(ImgEmbed, self).__init__(**kwargs)
        self.len = glyph_block_size
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.rate = rate
        
    def build(self, input_shape):
        n = input_shape[1] * input_shape[2] / (self.len ** 2)
        self.linear1 = layers.Dense(self.ff_dim, activation='relu', dtype=self.dtype)
        self.linear2 = layers.Dense(self.embed_dim, dtype=self.dtype)
        self.norm = layers.LayerNormalization()
        self.pos_enc = PosEnc2D(embed_dim=self.embed_dim, n=n)
        self.dropout = layers.Dropout(self.rate)

    def call(self, x, training=None):
        shape = tf.shape(x)
        x = tf.cast(x, self.dtype) / 255.0
        pad = [shape[1] % self.len, shape[2] % self.len]
        x = tf.pad(x, [[0, 0], [pad[0] // 2, pad[0] - pad[0] // 2], [pad[1] // 2, pad[1] - pad[1] // 2]])
        new_shape = [tf.shape(x)[1] // self.len, tf.shape(x)[2] // self.len]
        x = tf.reshape(x, [shape[0], new_shape[0], self.len, new_shape[1], self.len])
        x = tf.reshape(tf.transpose(x, [0,1,3,2,4]), [shape[0], new_shape[0], new_shape[1], self.len * self.len])
        
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.norm(x)
        p = self.pos_enc(x)[tf.newaxis,:,:,:]
        x = x + p
        x = tf.reshape(x, [tf.shape(x)[0], new_shape[0] * new_shape[1], tf.shape(x)[-1]])

        return self.dropout(x, training=training)
    
    def get_prunable_weights(self):
        return [self.linear1.kernel, self.linear2.kernel]
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "ff_dim": self.ff_dim, 'glyph_block_size': self.len, 'rate': self.rate})
        return config


class TokenEmbed(layers.Layer, tfmot.sparsity.keras.PrunableLayer):

    def __init__(self, vocab_size, expected_len, embed_dim=512, **kwargs):
        super(TokenEmbed, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.expected_len = expected_len
        
    def build(self, input_shape):
        self.token_embed = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim, mask_zero=True,
                                            embeddings_constraint=keras.constraints.MaxNorm(max_value=1, axis=-1),
                                            embeddings_regularizer=keras.regularizers.OrthogonalRegularizer())
        expected_len = self.expected_len
        self.pos_enc = PosEnc1D(embed_dim=self.embed_dim, n=expected_len)
        self.mask_layer = layers.Masking()
    
    def call(self, x):
        x = self.token_embed(x)
        mask = x._keras_mask
        x = self.mask_layer(x)
        x *= tf.sqrt(tf.cast(self.embed_dim, self.dtype))
        p = self.pos_enc(x)[tf.newaxis,:,:]
        return x + p, mask
    
    def get_prunable_weights(self):
        return []
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "vocab_size": self.vocab_size, 'expected_len': self.expected_len})
        return config


class EncoderBlock(layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, rate=0.1, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
    
    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(self.ff_dim, activation="relu"), layers.Dense(self.embed_dim)]
        )
        self.layernorms = [layers.LayerNormalization(epsilon=1e-6) for _ in range(2)]
        self.dropouts = [layers.Dropout(self.rate) for _ in range(2)]
    
    def call(self, value, training=None):

        out1 = self.att(value, value)
        out1 = self.dropouts[0](out1, training=training)
        out1 = self.layernorms[0](value + out1)
        
        out2 = self.ffn(out1)
        out2 = self.dropouts[1](out2, training=training)
        out2 = self.layernorms[1](out1 + out2)
        return out2
    
    def get_prunable_weights(self):
        att_weights = [w for w in self.att.trainable_weights if w.name[-8:-2]=='kernel']
        return att_weights + [self.ffn.layers[0].kernel, self.ffn.layers[1].kernel]
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, 'ff_dim': self.ff_dim, 'rate': self.rate})
        return config


class DecoderBlock(layers.Layer, tfmot.sparsity.keras.PrunableLayer):

    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, rate=0.1, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
    
    def build(self, input_shape):
        self.self_att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.mem_att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(self.ff_dim, activation="relu"), layers.Dense(self.embed_dim)]
        )
        self.layernorms = [layers.LayerNormalization(epsilon=1e-6) for _ in range(3)]
        self.dropouts = [layers.Dropout(self.rate) for _ in range(3)]

    def call(self, query, value, training=None, causal_mask=True, padding_mask=None):
        qdim = tf.shape(query)[1]
        if causal_mask:
            ij = tf.range(qdim)
            mask = ij[tf.newaxis,:] <= ij[:, tf.newaxis]
        else:
            mask = tf.ones((qdim,qdim), dtype='bool')
        self_mask = tf.logical_and(padding_mask[:,:,tf.newaxis], padding_mask[:,tf.newaxis,:])
        self_mask = tf.logical_and(self_mask, mask[tf.newaxis,:,:])

        out1 = self.self_att(query, query, attention_mask=self_mask)
        out1 = self.dropouts[0](out1, training=training)
        out1 = self.layernorms[0](query + out1)
        
        out2 = self.mem_att(out1, value, attention_mask=padding_mask[:,:,tf.newaxis])
        out2 = self.dropouts[1](out2, training=training)
        out2 = self.layernorms[1](out1 + out2)
        
        out3 = self.ffn(out2)
        out3 = self.dropouts[2](out3, training=training)
        out3 = self.layernorms[2](out2 + out3)
        return out3
    
    def get_prunable_weights(self):
        self_att_weights = [w for w in self.self_att.trainable_weights if w.name[-8:-2]=='kernel']
        mem_att_weights = [w for w in self.mem_att.trainable_weights if w.name[-8:-2]=='kernel']
        return self_att_weights + mem_att_weights + [self.ffn.layers[0].kernel, self.ffn.layers[1].kernel]
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, 'ff_dim': self.ff_dim, 'rate': self.rate})
        return config


def Cangjie(vocab_size, expected_len, encoder_stacks=6, decoder_stacks=6, num_heads=8, embed_dim=512, ff_dim=2048,
                 rate=0.1, glyph_block_size=16, image_dim=128, **kwargs):

    image = layers.Input(shape=(image_dim, image_dim), name='Image', dtype='uint8')
    code = layers.Input(shape=(None,), name='Code', dtype='uint8')

    img_embed = ImgEmbed(embed_dim=embed_dim, ff_dim=ff_dim, glyph_block_size=glyph_block_size, rate=rate, name="Image_Embedding")
    tkn_embed = TokenEmbed(vocab_size, expected_len, embed_dim=embed_dim, name="Token_Embedding")
    encoders = [EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, rate=rate, name=f"Encoder_{i+1}") for i in range(encoder_stacks)]
    decoders = [DecoderBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, rate=rate, name=f"Decoder_{i+1}") for i in range(decoder_stacks)]
    output_ffn = layers.Dense(vocab_size, activation='softmax', name='Reverse_Embedding')
    
    x = img_embed(image)
    for encoder in encoders:
        x = encoder(x)

    y, m = tkn_embed(code)
    for decoder in decoders:
        y = decoder(y, x, padding_mask=m)

    output = output_ffn(y)
    
    model = keras.Model(inputs=[image, code], outputs=output, **kwargs)
    return model

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

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)
