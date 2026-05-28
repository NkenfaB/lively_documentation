import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

# Load data
X_train = np.load('data/processed/features/X_train.npy')
y_train = np.load('data/processed/features/y_train.npy')
X_val = np.load('data/processed/features/X_val.npy')
y_val = np.load('data/processed/features/y_val.npy')

# Remap labels: 1->0 (COVID), 2->1 (HEALTHY)
label_map = {1: 0, 2: 1}
y_train = np.array([label_map[y] for y in y_train])
y_val = np.array([label_map[y] for y in y_val])

print(f"Training samples: {len(X_train)}")
print(f"  COVID: {np.sum(y_train == 0)} ({100*np.mean(y_train == 0):.1f}%)")
print(f"  HEALTHY: {np.sum(y_train == 1)} ({100*np.mean(y_train == 1):.1f}%)")

# Compute MODERATE class weights (capped to avoid extreme values)
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
# Cap the weights to prevent model collapse
max_weight = 3.0
class_weights_array = np.clip(class_weights_array, 0.5, max_weight)
class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}

print(f"\nClass weights (capped at {max_weight}):")
print(f"  COVID (class 0): {class_weights[0]:.2f}")
print(f"  HEALTHY (class 1): {class_weights[1]:.2f}\n")

# Build model with DROPOUT to prevent overfitting
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X_train.shape[1:]),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),  # Added dropout
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.4),  # Added dropout
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),  # Added dropout
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Use a LOWER learning rate for more stable training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with class weights and early stopping
checkpoint_dir = Path('models/checkpoints')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'models/checkpoints/baseline_cnn_improved.keras',
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,  # More epochs with early stopping
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training complete!")
print(f"Model saved to: models/checkpoints/baseline_cnn_improved.keras")
