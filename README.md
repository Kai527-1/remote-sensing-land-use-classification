# Remote Sensing Land Use Classification with EuroSAT

This project implements a remote sensing image classification pipeline using Python and PyTorch.  
It uses the EuroSAT RGB dataset to classify satellite image patches into 10 land-use categories.

## Overview

The goal of this project is to automatically classify satellite image patches into semantic land-use classes such as forest, river, residential area, and industrial area.

This project is designed as a clean baseline for learning and demonstrating:

- remote sensing image classification
- PyTorch data pipeline construction
- transfer learning with convolutional neural networks
- model evaluation using standard classification metrics

## Dataset

This project uses the **EuroSAT RGB** dataset.

The dataset contains 10 land-use / land-cover classes:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

Place the dataset under:

```text
data/EuroSAT_RGB
