# Content-Based Image Retrieval with ResNet-50 & Deep Metric Learning

This project implements a content-based image retrieval system in **PyTorch** using a pretrained **ResNet-50** backbone fine-tuned with deep metric learning. It benchmarks three embedding optimization strategies—**Contrastive Loss**, **Triplet Loss**, and **Hard-Negative Mining**—on the Caltech-101 dataset, evaluating retrieval quality via Recall@K and t-SNE feature space visualizations.


---

## Setup

Download Caltech-101 dataset and place it as:

```
Dataset/caltech-101/
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Subset Dataset Indices 

Run this command to make a consistent subset to use throughout project:

```
cd Codes
python data_subset.py --data_path ../Dataset/caltech-101
```

## Training

```
cd Codes
python train.py --data_path ../Dataset/caltech-101 --mode contrastive --epochs 5
python train.py --data_path ../Dataset/caltech-101 --mode triplet --epochs 5
python train.py --data_path ../Dataset/caltech-101 --mode hard --epochs 5
```

---

## Save Embeddings

```
cd Codes
python save_embeddings.py --data_path ../Dataset/caltech-101 --model_path ../Saved_Models/contrastive_best.pth --mode contrastive
python save_embeddings.py --data_path ../Dataset/caltech-101 --model_path ../Saved_Models/triplet_best.pth --mode triplet
python save_embeddings.py --data_path ../Dataset/caltech-101 --model_path ../Saved_Models/hard_best.pth --mode hard
```

---

## Retrieval Evaluation

```
cd Codes
python retrieval.py --data_path ../Dataset/caltech-101 --mode contrastive
python retrieval.py --data_path ../Dataset/caltech-101 --mode triplet
python retrieval.py --data_path ../Dataset/caltech-101 --mode hard
```

Outputs:

- Recall@1, Recall@5
- Retrieval visualization
- t-SNE plots

---

## Inference

Generate embedding for new images:

```
cd Codes
python inference.py --model_path ../Saved_Models/contrastive_best.pth --image_path ../Dataset/caltech-101/airplanes/image_0001.jpg
python inference.py --model_path ../Saved_Models/triplet_best.pth --image_path ../Dataset/caltech-101/airplanes/image_0001.jpg
python inference.py --model_path ../Saved_Models/hard_best.pth --image_path ../Dataset/caltech-101/airplanes/image_0001.jpg
```
For checking embeddings of two different images, along with cosine score:

```
cd Codes
python inference.py --model_path ../Saved_Models/contrastive_best.pth --image_path ../Dataset/caltech-101/airplanes/image_0001.jpg ../Dataset/caltech-101/airplanes/image_0002.jpg 
python inference.py --model_path ../Saved_Models/triplet_best.pth --image_path ../Dataset/caltech-101/airplanes/image_0001.jpg ../Dataset/caltech-101/airplanes/image_0002.jpg 
python inference.py --model_path ../Saved_Models/hard_best.pth --image_path ../Dataset/caltech-101/airplanes/image_0001.jpg ../Dataset/caltech-101/airplanes/image_0002.jpg 
```
---
