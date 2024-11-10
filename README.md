# Naive Bayes Classifier from Scratch

This project implements a Naive Bayes classifier from scratch using Python, applied to the Iris dataset for classification.

## Project Structure

- `data/iris.csv`: Iris dataset
- `src/naive_bayes.py`: Naive Bayes implementation
- `README.md`: Project instructions
- `requirements.txt`: Dependencies

## Overview

Naive Bayes is a probabilistic classifier based on Bayes' Theorem, assuming independence among features. This implementation uses Gaussian distribution to estimate probabilities for continuous features in the Iris dataset.

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/Setsofia-lab/naivebayes.git
cd naive-bayes-iris
pip install -r requirements.txt
```

## Usage

1. **Train the Classifier**:

    ```python
    from src.naive_bayes import NaiveBayesClassifier
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    ```

2. **Make Predictions**:

    ```python
    predictions = nb.predict(X_test)
    ```

3. **Evaluate Accuracy**:

    ```python
    accuracy = nb.evaluate(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    ```


