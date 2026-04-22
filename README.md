# Fruit Grading System Using Deep Learning

An Fruit Grading System that automatically classifies fruits based on quality using Deep Learning models such as EfficientNet, ConvNeXt, and Swin Transformer. This project helps automate fruit quality assessment into categories like Good, Bad, and Mixed, reducing manual labor and increasing accuracy.

## Project Overview

Traditional fruit grading is done manually, which is:

- Time-consuming  
- Error-prone  
- Inconsistent  
- Expensive for large-scale farms  

This project solves these problems using Computer Vision and Deep Learning by analyzing fruit images and predicting:

- Fruit Type  
- Fruit Quality  
- Confidence Score  

## Features

- Automatic Fruit Classification  
- Quality Detection (Good / Bad / Mixed)  
- Multiple Deep Learning Models  
- Ensemble Prediction System  
- Unknown Fruit Detection  
- High Accuracy Results   

## Models Used

| Model | Purpose |
|------|---------|
| EfficientNet | Lightweight and High Accuracy |
| ConvNeXt | Modern CNN Architecture |
| Swin Transformer | Vision Transformer Model |

## Dataset Structure

```bash
dataset/
├── Apple/
├── Banana/
├── Grape/
├── Guava/
├── Lime/
├── Mango/
├── Orange/
└── Pomegranate/

quality/
├── Good/
├── Bad/
└── Mixed/

Dataset URL : https://www.kaggle.com/datasets/dhanushkodi12/fruits
Models URL : https://www.kaggle.com/datasets/dhanushkodi12/models
