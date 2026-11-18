# Cats vs Dogs Image Classification (CNN vs Transfer Learning)

A binary image-classification project that compares a small convolutional neural network trained from scratch with a transfer-learning model built on MobileNetV2, using TensorFlow and Keras.

## Overview

The notebook builds a complete deep-learning workflow: downloading the dataset, filtering corrupted files, creating train/validation/test splits, applying data augmentation, training two models, and evaluating both on a held-out test set with classification reports and confusion matrices.

The main finding is the gap between the two approaches on a small dataset: the model trained from scratch stayed close to chance level, while the pre-trained MobileNetV2 reached about 95% test accuracy.

## Dataset

- **Source:** Microsoft Cats and Dogs dataset (`kagglecatsanddogs_5340.zip`), downloaded in the notebook.
- **Cleaning:** image files smaller than 1 KB (empty or corrupted) are filtered out.
- **Subset used:** 1,000 images per class (2,000 in total), chosen at random with `random.seed(42)` to keep training fast.
- **Split (80/10/10):** 1,600 training, 200 validation and 200 test images (100 cats and 100 dogs in the test set).

## Methodology

- **Preprocessing:** images resized to 150×150 and rescaled to the 0-1 range.
- **Augmentation (training set only):** rotation up to 15°, width and height shifts of 10%, horizontal flip.
- **Training setup:** batch size 32, Adam optimizer, binary cross-entropy loss, 5 epochs per model.

### Model 1: Custom CNN (from scratch)

Two convolution blocks (32 and 64 filters, each with batch normalization and max pooling), followed by Flatten, a Dense layer with 64 units, Dropout (0.5) and a sigmoid output.

### Model 2: Transfer learning (MobileNetV2)

MobileNetV2 with ImageNet weights and the convolutional base frozen, followed by global average pooling, a Dense layer with 64 units, Dropout (0.3) and a sigmoid output. Only the new layers are trained; no fine-tuning of the base.

## Results

Evaluated on the 200 held-out test images (final 5-epoch run in the notebook):

| Model | Test accuracy | Cats recall | Dogs recall |
|---|---|---|---|
| Custom CNN | 0.54 | 0.97 | 0.11 |
| MobileNetV2 (transfer learning) | 0.95 | 0.92 | 0.98 |

An earlier 3-epoch run saved in the same notebook gave 0.51 for the custom CNN and 0.97 for MobileNetV2, so results vary slightly between runs.

The custom CNN labelled almost every image as a cat, so its accuracy is close to the 50% expected from guessing. The transfer-learning model was accurate for both classes.

## Limitations

- Only 2,000 images were used, and the test set has just 200 images, so the accuracy figures carry noticeable uncertainty.
- Each model was trained for 5 epochs with no hyperparameter tuning. The weak result for the custom CNN is probably due to the small dataset and short training, but this was not investigated further.
- The MobileNetV2 base was kept frozen; fine-tuning was not tested.
- Results come from a single random split.

## Future Improvements

- Use the full dataset (about 25,000 images) and train longer.
- Fine-tune the top layers of MobileNetV2.
- Add training curves, confusion-matrix images and ROC curves to the repository.
- Try cross-validation to measure variation between runs.

## Project Files

| File | Description |
|---|---|
| `Image_Classification.ipynb` | Main notebook: data preparation, training and evaluation |
| `image_classification.py` | Python export of the notebook |

## How to Run

The notebook was built in Google Colab and downloads the dataset from the internet. A GPU runtime makes training faster.

1. Open the notebook in Google Colab and select a GPU runtime.
2. Run all cells in order.

To run locally:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Image_Classification.ipynb
```

## Technologies

Python · TensorFlow / Keras · MobileNetV2 · scikit-learn · Matplotlib · Seaborn · Google Colab

## Author

**Alisha Arif** · [GitHub](https://github.com/alishaarif500) · [LinkedIn](https://linkedin.com/in/alisha-arif-22aa42386/)
