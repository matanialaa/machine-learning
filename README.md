[README.md](https://github.com/user-attachments/files/33675778/README.md)
# Machine Learning Coursework

**Classification and regression on a medical dataset, implemented in Python, with several algorithms written from scratch.**

Three assignments covering the full workflow: data exploration and preparation, building classifiers, tuning hyperparameters, and comparing regression models.

## Results at a Glance

| Task | Result |
|---|---|
| k-Nearest Neighbors (from scratch) | Test accuracy raised from **58.8% to 86.0%** through scaling and tuning |
| Decision tree tuning | Best of a **361-combination** grid search |
| Lasso regression | **89% lower** validation error than the baseline |
| Gradient boosting regression | **98% lower** validation error than the baseline |

## Part 1 – Data Exploration and k-NN

- Explored a **1,250-patient, 25-feature** dataset with histograms, pair plots, joint plots, and 3D plots.
- Found outliers in household income and filled missing values with the **median**, which is less sensitive to outliers than the mean.
- Built a **data preparation pipeline** that fits scaling only on the training set, to avoid data leakage.
- Implemented a **vectorized k-NN classifier** from scratch with SciPy `cdist` and NumPy `argpartition`.

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| 1-NN, raw features | 100.0% | 58.8% |
| 1-NN, normalized | 100.0% | 75.6% |
| 5-NN, normalized | 84.3% | 85.2% |
| 9-NN, tuned with cross-validation | – | **86.0%** |

![k-NN decision boundaries](images/knn_boundaries.png)

## Part 2 – Model Tuning and Linear Classifiers

- Picked the best `k` for k-NN with cross-validation and identified overfitting (k = 1) and underfitting (very large k).
- Ran a **361-combination grid search** over decision tree depth and minimum leaf size.
- Wrote a **gradient-descent linear classifier** and checked its gradients against numerical approximations.
- Analyzed how the learning rate and regularization strength affect loss and accuracy during training.

## Part 3 – Regression

| Model | Validation MSE |
|---|---|
| Dummy baseline | 0.0704 |
| Linear regression | 0.0096 |
| Lasso regression | 0.0078 |
| Gradient boosting | **0.0017** |

- Implemented the gradient for linear regression and tuned the learning rate.
- Tuned Lasso's alpha and used its coefficients to find the most important features.
- Compared Lasso and Ridge, and explained why normalization matters for regularized models.

## Tech Stack

Python · NumPy · pandas · scikit-learn · SciPy · Matplotlib · Seaborn · Jupyter

## Repository Structure

```
hw1-data-exploration-knn/
hw2-model-tuning/
hw3-regression/
images/
```

## About

Coursework for Introduction to Machine Learning (236756) at the Technion, completed with a partner.


