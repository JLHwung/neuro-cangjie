from models import *

MAX_LEN = 64
VOCAB_SIZE = 30

model = Cangjie(VOCAB_SIZE, MAX_LEN)
status = model.load_weights('cangjie_pruned/variables/variables')
status.assert_consumed()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save the model.
with open('cangjie.tflite', 'wb') as f:
    f.write(tflite_model)
