import numpy as np 


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_weights(logits, weight_cap=None):
    weights = np.exp(logits)
    probas = sigmoid(logits)
    if weight_cap is not None:
        weights = np.clip(weights, 0, weight_cap)
    return weights, probas