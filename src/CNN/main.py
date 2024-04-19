import tensorflow as tf
from __init__ import load_model

model = load_model()
#model.export("saved_model")

# Convert the model
model.summary()
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model") # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]

#converter.target_spec.supported_ops = [
#    tf.lite.OpsSet.TFLITE_BUILTINS,
#    tf.lite.OpsSet.SELECT_TF_OPS
#]
#converter._experimental_lower_tensor_list_ops = False
#converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()

# Save the model.
with open('smaller_optimized_model.tflite', 'wb') as f:
  f.write(tflite_model)

#model.save("keras_model.h5", overwrite=True)