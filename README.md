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
├─ train.py
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

## Example Results

> Replace placeholders with actual outputs

| Input | Ground Truth | Prediction |  
|-------|--------------|------------|  
| ![input](output/input_placeholder.jpg) | ![target](output/target_placeholder.jpg) | ![pred](output/pred_placeholder.jpg) |

---

## Notes

- Discriminator updates only if loss > 0.5  
- Images normalized to [-1, 1]  
- Random augmentations applied during training  
- Small dataset and GPU limits affected final quality  
- With more data and stronger hardware, model could effectively clean and digitize scanned manga pages
