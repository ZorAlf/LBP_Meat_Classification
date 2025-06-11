# LBP Klasifikasi Daging (Kuda, Sapi, dan Babi)

This project implements a meat classification system to distinguish between horse, beef, and pork using **Local Binary Pattern (LBP)** for feature extraction and **Random Forest** for classification. The dataset used contains images of horse, beef, and pork meat, which is processed to extract texture-based features for accurate classification.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [How It Works](#how-it-works)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
The project uses Local Binary Pattern (LBP) to extract texture features from meat images and a Random Forest classifier to predict the meat type (horse, beef, or pork). It includes functionality to:
- Extract LBP features from grayscale images.
- Train a Random Forest model.
- Visualize results with confusion matrices and LBP-transformed images.
- Predict the class of new images with confidence scores.

## Dataset
The dataset is sourced from Kaggle: [Pork Meat and Horse Meat Dataset](https://www.kaggle.com/datasets/iqbalagistany/pork-meat-and-horse-meat-dataset). It contains images of:
- Horse meat
- Beef
- Pork

**Note**: Ensure the dataset is organized in a folder structure where each class (e.g., `horse`, `beef`, `pork`) is a subdirectory containing the respective images.

## Requirements
- Python 3.7+
- Libraries:
  - `opencv-python`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `scikit-image`
  - `matplotlib`
  - `seaborn`
  - `google-colab` (if running on Google Colab)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/LBP_Meat_Classification.git
   cd LBP_Meat_Classification
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/iqbalagistany/pork-meat-and-horse-meat-dataset) and place it in a folder (e.g., `dataset/`).

## Usage
1. Update the `DATASET_PATH` in the script to point to your dataset folder:
   ```python
   DATASET_PATH = "/path/to/your/dataset"
   ```
2. Run the script:
   ```bash
   python lbp_meat_classification.py
   ```
3. The script will:
   - Extract LBP features from the dataset.
   - Train a Random Forest classifier.
   - Display a classification report and confusion matrix.
   - Allow you to upload new images for prediction (in Colab) with visualizations.

## Code Structure
- `lbp_meat_classification.py`: Main script containing:
  - LBP feature extraction (`extract_lbp_features`).
  - Dataset loading and feature preparation (`load_dataset_features`).
  - Model training with visualization (`train_lbp_classifier`).
  - Prediction and visualization for new images (`show_prediction_with_lbp`).
- `dataset/`: Folder for storing the dataset (not included in the repository).

## How It Works
1. **Feature Extraction**: Images are converted to grayscale, and LBP is applied to extract texture features. The LBP histogram is normalized to create a feature vector.
2. **Training**: The feature vectors and labels are split into training and test sets. A Random Forest classifier is trained on the training data.
3. **Evaluation**: The model is evaluated using a classification report and a confusion matrix.
4. **Prediction**: New images are processed similarly, and the model predicts the meat type with confidence scores. Visualizations include the original image, grayscale, and LBP image.

## Results
The model provides:
- A classification report with precision, recall, and F1-score for each class.
- A confusion matrix visualizing prediction performance.
- For new images, it outputs:
  - Predicted class and confidence score.
  - Visualizations of the original, grayscale, and LBP images.

Example output for a prediction:
```
🔎 Prediksi:
 - Beef: 10.00%
 - Horse: 15.00%
 - Pork: 75.00%

✅ Prediksi Teratas: Pork (75.00%)
```

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.