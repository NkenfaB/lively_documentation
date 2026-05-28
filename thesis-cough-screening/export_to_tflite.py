"""Export trained model to TFLite - Run this on RunPod"""
import tensorflow as tf
from pathlib import Path

# Load the trained model
model_path = 'models/checkpoints/baseline_cnn.keras'
model = tf.keras.models.load_model(model_path)

print(f"Loaded model from {model_path}")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
output_dir = Path('models/exported')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'baseline_cnn.tflite'

output_path.write_bytes(tflite_model)

# Print file sizes for comparison
keras_size = Path(model_path).stat().st_size / 1024
tflite_size = len(tflite_model) / 1024

print(f"\n✓ Successfully exported to TFLite!")
print(f"  Keras model:  {keras_size:.1f} KB")
print(f"  TFLite model: {tflite_size:.1f} KB ({output_path})")
