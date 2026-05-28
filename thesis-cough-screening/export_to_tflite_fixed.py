"""Export trained model to TFLite with batch norm fix"""
import tensorflow as tf
from pathlib import Path
import numpy as np

# Load the trained model
model_path = 'models/checkpoints/baseline_cnn.keras'
model = tf.keras.models.load_model(model_path)

print(f"Loaded model from {model_path}")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Convert to TFLite with optimizations to avoid batch norm issues
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Use default optimizations to handle batch normalization properly
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide representative dataset for quantization
X_test = np.load('data/processed/features/X_test.npy')
def representative_dataset():
    for i in range(min(100, len(X_test))):
        yield [X_test[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset

try:
    tflite_model = converter.convert()
    print("✓ Conversion successful with quantization")
except Exception as e:
    print(f"Quantization failed: {e}")
    print("Trying without quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    print("✓ Conversion successful without quantization")

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
