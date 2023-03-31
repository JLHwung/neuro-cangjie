from models import *

data = tf.data.Dataset.load('compiled_data', compression='GZIP')

train_size = int(0.9 * len(data))
train_data = data.take(train_size)
test_data = data.skip(train_size)


def preprocess(dataset, batch_size):
    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

IMG_SIZE = 128
MAX_LEN = 64
VOCAB_SIZE = 30
BATCH_SIZE = 128

train_data = preprocess(train_data, batch_size=BATCH_SIZE)
test_data = preprocess(test_data, batch_size=BATCH_SIZE)

model = Cangjie(VOCAB_SIZE, MAX_LEN)
status = model.load_weights('cangjie/variables/variables')
status.assert_consumed()

# +
EPOCHS = 20
end_step = EPOCHS * len(train_data)

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.00,
                                                   final_sparsity=0.95,
                                                   begin_step=0,
                                                   end_step=end_step,
                                                   frequency=len(train_data) // 10)
}

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
# -
pruned_model.compile(keras.optimizers.Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(ignore_class=0, label_smoothing=0.01), metrics=[SparseCategoricalAccuracy(ignore_class=0)], jit_compile=False)

pruned_model.summary()

# ## Training

log_dir = './logs/cangjie_pruned'

pruned_model.fit(train_data, epochs=EPOCHS, validation_data=test_data, callbacks=[
    keras.callbacks.TensorBoard(log_dir=log_dir), tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)])

cangjie = tfmot.sparsity.keras.strip_pruning(pruned_model)
cangjie.save('cangjie_pruned', save_format='tf')