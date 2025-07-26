# Malt Quality Hyperspectral Imaging Prediction and Classification

This project aims to predict and classify malt quality parameters of barley seeds using hyperspectral reflectance data. Precition and classification are performed both at the seed and variety level using traditional ML models and a 1D-CNN.

## Description

The repository includes image processing, spectral feature extraction, dataset creation, classical ML and CNN modeling, as well as results analysis and visualization.

## Pipeline Overview

| Step | Folder | Description |
|------|--------|-------------|
| **1** | `01_image_processing_and_feature_extraction/` | Image segmentation and spectral feature extraction including index generation |
| **2** | `02_target_preparation/` | Creation of classification labels and expansion to seed level |
| **3** | `03_dataset_selection/` | Dataset versioning and optimal dataset selection |
| **4** | `04_feature_selection_and_ml_modeling/` | Grid search and ML pipeline tuning (SVM, RF, etc.) |
| **5** | `05_cnn_modeling/` | Deep learning models (1D CNN for prediction and classification) |
| **6** | `06_results_analysis_and_visualization/` | Evaluation, statistics, and performance plots |

NOTE: In order to run stage 1 of the pipeline you need the raw image data, which are provided separately from this repository because the high quality hyperspectral images requires a big amount of storage.

If you don't have access to the raw data you can run the code starting from stage 2, together with the excel files that are present in the `dataset/` folder.

The files in the the `dataset/` folder are the output of the scripts in stage 1, which are preprocessed data from the raw images.

## Dependencies

You can install required packages with:

```bash
pip install -r requirements.txt
```