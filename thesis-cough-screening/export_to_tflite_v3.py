"""Export trained model to TFLite - workaround for batch norm bug"""
import tensorflow as tf
from pathlib import Path
import numpy as np

# Load the trained model
model_path = 'models/checkpoints/baseline_cnn.keras'
model = tf.keras.models.load_model(model_path)

print(f"Loaded model from {model_path}")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Create a concrete function - this bakes in the batch norm
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 64, 256, 1], dtype=tf.float32)])
def model_predict(x):
    return model(x, training=False)

concrete_func = model_predict.get_concrete_function()

# Convert using concrete function instead of keras model
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

# Save TFLite model
output_dir = Path('models/exported')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'baseline_cnn.tflite'

output_path.write_bytes(tflite_model)

# Print file sizes
keras_size = Path(model_path).stat().st_size / 1024
tflite_size = len(tflite_model) / 1024

print(f"\n✓ Successfully exported to TFLite!")
print(f"  Keras model:  {keras_size:.1f} KB")
print(f"  TFLite model: {tflite_size:.1f} KB")
print(f"  Saved to: {output_path}")
