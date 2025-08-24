# Pix2Pix & CycleGAN Manga Cleanup
This repository implements an image-to-image translation model combining Pix2Pix and CycleGAN concepts using TensorFlow and Keras. It supports training and inference on paired datasets, with Discord integration for live training updates.
One of my first ML projects. The goal was to clean up scanned manga images and make them look digitally converted. Results are limited due to a small dataset and limited compute (trained on an NVIDIA 3060 12GB), but with more data and stronger hardware, the approach is viable.

---

## Features

- U-Net generator and PatchGAN discriminator  
- Image preprocessing: resize, normalize, random jitter, horizontal flip  
- L1 + GAN loss training  
- Discord webhook integration for live epoch updates  
- Saves model checkpoints and generated images

---

## Dataset & Pre-trained Weights

- Dataset (Kaggle): [Pix2Pix Manga Dataset](https://www.kaggle.com/datasets/fghjkgcfxx56u/pix2pixmanga)  
- Pre-trained Weights (Kaggle): [Model Weights](https://www.kaggle.com/datasets/fghjkgcfxx56u/model-weights)  

> Place pre-trained weights in `weights/gen/` and `weights/disc/` to skip training.

---

## Requirements

Python version: 3.8+  
Libraries:

```bash
pip install tensorflow keras discord matplotlib tqdm numpy
```

---

## Project Structure

```text
.
├─ weights/
│  ├─ gen/
│  └─ disc/
├─ output/
├─ model.py
└─ data/
   ├─ train/
   └─ val/
```

---

## Training

1. Prepare paired images: left = target (clean digital look), right = input (scanned manga)  

2. Update dataset paths in `train.py`:

```python
train_dataset = tf.data.Dataset.list_files("../input/roombav2/train/*.jpg")
test_dataset = tf.data.Dataset.list_files("../input/roombav2/val/*.jpg")
```

3. Run training:

```bash
python train.py
```

- Checkpoints saved in `weights/gen/` and `weights/disc/`  
- Generated images saved per epoch in `output/`  
- Discord webhook posts epoch updates and sample outputs  

---

## Inference

```python
gen = generator()
gen.load_weights("weights/gen/gen_checkpoint")
prediction = gen(input_image, training=False)
```

---

## Results

> Some of the best results from the model

![img](https://storage.googleapis.com/kaggle-script-versions/94500343/output/output/epoch_1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250824%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250824T144259Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=3c8750caf58177369bdc8fc16a54540b7956f74ef3588b76dffbc26e5fc12f6d5f33bd1a28c58f214d2f79883352d7e69142a4a86a4f58aaf67f3de17fc5a8d84a1f7e47d88a2530f9054508fe20c09f62355d3dd37713f67511c2171e179dbe2d960b4de238e2a509e8248c6f533dfe06157b28d69d0c36d57e3c9c4887717ace8af781b60bbcfd340de849766991ee78e957e7aa9d555313eadca4ae27bb2543caeb0789d6b44ee57f4da47854a2c1f070a06b2472b0907b27b8117ba40fdbdca03bfa8c3e8d9544357762c43d884c884994db74be088ec97351a1c2a3adc2bf8b2818876c745835bb3d9067d655565d03513ea75d642e1abbff33ca506e8d)

---

## Notes

- Discriminator updates only if loss > 0.5  
- Images normalized to [-1, 1]  
- Random augmentations applied during training  
- Small dataset and GPU limits affected final quality  
- With more data and stronger hardware, model could effectively clean and digitize scanned manga pages
