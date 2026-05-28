import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Load model
model = tf.keras.models.load_model('models/checkpoints/baseline_cnn_weighted.keras')

# Load test data
X_test = np.load('data/processed/features/X_test.npy')
y_test = np.load('data/processed/features/y_test.npy')

# Remap labels
label_map = {1: 0, 2: 1}
y_test_mapped = np.array([label_map[label] for label in y_test])

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test_mapped, verbose=1)

# Get predictions
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n" + "="*60)
print("COMPARISON: Baseline vs Class-Weighted Model")
print("="*60)
print("\nBASELINE MODEL (previous):")
print("  Test Accuracy: 74.93%")
print("  COVID Recall: 21.1% (43/204)")
print("  HEALTHY Recall: 94.9% (522/550)")

print("\nCLASS-WEIGHTED MODEL (new):")
print(f"  Test Accuracy: {test_acc*100:.2f}%")
print(classification_report(y_test_mapped, y_pred_classes, target_names=['COVID', 'HEALTHY']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_mapped, y_pred_classes))
print("\n[COVID predictions on rows, HEALTHY on columns]")
