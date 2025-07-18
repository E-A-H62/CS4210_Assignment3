# Perceptron vs MLP Classifier on Optical Digits Dataset

## Description

This project compares the performance of two supervised learning models—a single-layer Perceptron and a Multi-Layer Perceptron (MLP)—on the Optical Recognition of Handwritten Digits dataset (`optdigits.tra` and `optdigits.tes`).
The models are trained and tested under various combinations of hyperparameters including learning rate and whether or not to shuffle the training data. The program tracks and prints the highest accuracy achieved by each model across all tested configurations.

---

## Features

- Reads and processes digit recognition datasets using `pandas` and `numpy`
- Trains a Perceptron and an MLPClassifier from `scikit-learn`
- Varies hyperparameters:
  - Learning rate: `0.0001` to `1.0`
  - Shuffle: `True` or `False`
- Evaluates model performance on a test dataset
- Prints out the best performing configuration for each model

---

## Algorithms Used

### 1. **Perceptron**
- `eta0`: Learning rate
- `shuffle`: Whether to shuffle training data each epoch
- `max_iter`: 1000

### 2. **MLPClassifier (Multi-layer Perceptron)**
- Activation function: `logistic`
- `hidden_layer_sizes`: One hidden layer with 25 neurons
- `learning_rate_init`: Initial learning rate
- `shuffle`: Whether to shuffle training data each epoch
- `max_iter`: 1000

---

## Files

- `perceptron.py`: Main script that performs training, evaluation, and comparison
- `optdigits.tra`: Training dataset (CSV format)
- `optdigits.tes`: Testing dataset (CSV format)

---

## Example Output

```

Highest Perceptron accuracy so far: 0.91, Parameters: learning rate=0.01, shuffle=True
Highest MLP accuracy so far: 0.97, Parameters: learning rate=0.1, shuffle=False

````

---

## Requirements

- Python 3.x
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)

> Make sure to install the required libraries if not already installed:

```bash
pip install numpy pandas scikit-learn
````

---

## How to Run

Ensure `optdigits.tra` and `optdigits.tes` are in the same directory as `perceptron.py`, then run:

```bash
python perceptron.py
```
