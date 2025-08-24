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

![img](https://storage.googleapis.com/kaggle-script-versions/94500343/output/output/epoch_112.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250824%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250824T144259Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=64290190846f663111955c5c9a81e84f444499b1b2c36d980fedc829e32cad0dea91be0e33e9de6a1133ca33162b6c0d0537ba0220ccfd1a003e309a79bf539f7ce6866ec69a5fc0601a120312f6978328d40e7d42395559dfb3e78767f9f13cc06bc8f616158b31799c7c4034be837ca6e8b8ec15fc0bb337122639d62aa9cbac99d8ccb55c571522930058c51cef005780195fa8833d02c093b1af865b8a1c8796731a8bfe8e75f4fafa9021ea177fa950de868ddf24079afb9003df4874b3ba4290a391adde5b6fecaf04b4d1f35f0474f83188eda9c7f23aa3b3af35afbee3a85661e6b7a9116982bbd1d4e9cde86a67a3ec09d49ae13ff5edde3d9e08a2)

![img](https://storage.googleapis.com/kaggle-script-versions/94500343/output/output/epoch_12.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250824%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250824T144259Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=6b7f86b2420dc2180e88aa74ce5eb91938aa729cbb083a5834d7e7ef2eb45eeef65e5bd5d846b0c38cc003116031913618ac9a87fdb961ddee2c2b1b07da8543f725a0c6b3001a6e0c9797265b859499f50141d2762adbfe1ac98cbeb34e94d80a590af506092ed97eb9ce633679c87e7ac7b66fa0cf0ef504e2bea1db64bb8664bcb60df3b630bad330f1a835a8a7e02e2df3640c6815b3e09e3beb910d14431a903f63143b31b81246a3903c89a2a30df219319b3addd491d9ead410c428d4aebbe486efeed5d0e93704e5327afabeb1b260e2d73822454b6ad5aa28708553812aee82a942dc99c3f856518b5d78b40a277738440b9e13f8d35e8e40088ce5)

---

## Notes

- Discriminator updates only if loss > 0.5  
- Images normalized to [-1, 1]  
- Random augmentations applied during training  
- Small dataset and GPU limits affected final quality  
- With more data and stronger hardware, model could effectively clean and digitize scanned manga pages
