# RunPod — MobileNetV2 transfer-learning run instructions

## Environment confirmed

| Item | Value |
|---|---|
| RunPod workspace | `/workspace/lively` |
| Python | `.venv/bin/python3` (Python 3.11) |
| TensorFlow | 2.16.2 |
| GPU | None on this pod — CPU training only |
| Feature arrays | `data/processed/features/` — all 6 .npy files present |

---

## What was done (completed session)

### Step 1 — Confirmed feature shapes

```
X_train: (4308, 64, 256, 1)   y_train: [COVID=991, HEALTHY=3317]
X_val:   (754,  64, 256, 1)   y_val:   [COVID=204, HEALTHY=550]
X_test:  (754,  64, 256, 1)   y_test:  [COVID=204, HEALTHY=550]
```

Labels in .npy files use the original 3-class schema (COVID=1, HEALTHY=2).
The training script remaps these to binary (COVID=0, HEALTHY=1) automatically.

### Step 2 — First training attempt (failed to export)

The first attempt used `tf.keras.layers.Lambda(mobilenet_v2.preprocess_input)`.
Keras 3 cannot serialize anonymous Python functions → `load_model()` raised
`TypeError: Could not locate function 'preprocess_input'` → TFLite converter
crashed with `LLVM ERROR: Failed to infer result type(s)`.

### Step 3 — Fixed architecture (no Lambda layer)

Replaced Lambda with two standard Rescaling layers:
```python
x = tf.keras.layers.Rescaling(255.0, name="scale_up")(x)
x = tf.keras.layers.Rescaling(1/127.5, offset=-1, name="mobilenet_norm")(x)
```
Mathematically identical to `preprocess_input`. Fully Keras-serializable.

### Step 4 — Training with class weights

```
Class weights: COVID=2.172, HEALTHY=0.648
Phase 1: backbone frozen, head only, Adam LR=1e-3, 10 epochs
Phase 2: top 30 backbone layers unfrozen (BN frozen), Adam LR=1e-5, 5 epochs
Batch size: 16
```

### Step 5 — Results

Threshold sweep: 0.25, 0.30, **0.35**, 0.40, 0.45, 0.50
Best threshold by COVID F1: **0.35**

| Metric | Baseline (threshold 0.35) | MobileNetV2 (threshold 0.35) |
|---|---|---|
| Accuracy | 0.6724 | 0.6711 |
| COVID precision | 0.4201 | 0.4197 |
| **COVID recall** | 0.5539 | **0.5637** |
| **COVID F1** | 0.4778 | **0.4812** |
| Healthy recall | 0.7164 | 0.7109 |

### Step 6 — TFLite export (fixed path)

Direct Keras→TFLite conversion crashes in TF 2.16 with MobileNetV2.
Fix: export to SavedModel first, then convert:

```python
model.export('models/exported/mobilenetv2_screening_savedmodel')
converter = tf.lite.TFLiteConverter.from_saved_model(
    'models/exported/mobilenetv2_screening_savedmodel')
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()
Path('models/exported/mobilenetv2_screening.tflite').write_bytes(tflite_model)
```

Output verified with TFLite interpreter — input/output shapes confirmed, single
sample inference produced valid probabilities.

---

## Artefacts produced

```
models/checkpoints/mobilenetv2_screening.keras
models/checkpoints/mobilenetv2_transfer_config.json
models/exported/mobilenetv2_screening_savedmodel/
models/exported/mobilenetv2_screening.tflite
reports/tables/mobilenetv2_screening_metrics.json
```

---

## Next steps

### On RunPod — download the TFLite file

```bash
# Option 1: scp from local
scp -i ~/.ssh/id_ed25519 \
  8n66wmww5cpavu-64411280@ssh.runpod.io:/workspace/lively/models/exported/mobilenetv2_screening.tflite \
  /Users/tm/Desktop/THESIS/lively/assets/model/mobilenetv2_screening.tflite
```

### In the lively/ app — integrate the new model

1. Copy `mobilenetv2_screening.tflite` → `lively/assets/model/`.
2. Update `lively/src/ml/modelAsset.ts` to reference the new filename.
3. Update the COVID decision threshold from `0.5` → `0.35` in the inference logic.
4. The preprocessing in `audioPreprocess.ts` does NOT need to change — the model
   accepts raw `(64, 256, 1)` float32 spectrograms and handles resize/norm internally.
5. Rebuild and test on device.

### If re-running training from scratch

```bash
cd /workspace/lively
.venv/bin/python3 -m src.models.train_mobilenetv2_transfer \
  --epochs 10 --fine-tune-epochs 5 --batch-size 16 --threshold 0.35
```

Note: `train_mobilenetv2_transfer.py` uses `tf.keras.layers.Lambda` for preprocessing —
the production version that was actually trained and exported uses the fixed
Rescaling-based architecture in the inline script above. The module-level script
should be updated to match before re-running.
