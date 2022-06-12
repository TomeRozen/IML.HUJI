from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    X_folds, y_folds = split_to_folds(X, y, cv)
    train_score_arr = []
    validation_score_arr = []
    for i in range(cv):
        X_train = concat_drop(cv, i, X_folds)
        y_train = concat_drop(cv, i, y_folds)
        X_valid, y_valid = X_folds[i], y_folds[i]

        estimator.fit(X_train, y_train)
        train_score_arr.append(scoring(y_train, estimator.predict(X_train)))
        validation_score_arr.append(scoring(y_valid, estimator.predict(X_valid)))

    return np.mean(train_score_arr), np.mean(validation_score_arr)


def split_to_folds(X, y, k):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split_idx = np.array_split(idx, k)
    X_folds = [X[indexes] for indexes in split_idx]
    y_folds = [y[indexes] for indexes in split_idx]
    return X_folds, y_folds


def concat_drop(cv, i, folds):
    return np.concatenate(
        [folds[idx] for idx in (np.delete(np.arange(cv), i))],
        axis=0)

