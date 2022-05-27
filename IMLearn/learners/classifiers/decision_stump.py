from __future__ import annotations

import math
import time
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_thr_err = math.inf
        threshold = None
        best_feature = -1
        best_direction = 0

        for feature_index in range(X.shape[1]):
            for direction in [-1, 1]:
                thr, thr_err = self._find_threshold(X[:, feature_index], y, direction)
                if thr_err < min_thr_err:
                    min_thr_err = thr_err
                    best_feature = feature_index
                    best_direction = direction
                    threshold = thr

        self.threshold_ = threshold
        self.j_ = best_feature
        self.sign_ = best_direction

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y_hat = np.sign(
            self.sign_ * (X[:, self.j_] - self.threshold_))
        y_hat[y_hat == 0] = self.sign_
        return y_hat

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        indexes_sorted = np.argsort(values)
        sort_values, sort_labels = values[indexes_sorted], labels[indexes_sorted]
        err = np.sum(np.abs(sort_labels)[np.sign(sort_labels) == sign])
        loss_array = np.append(err, err - np.cumsum(sort_labels * sign))

        min_index = np.argmin(loss_array)
        sort_values = np.append([-np.inf], sort_values)

        return sort_values[min_index], loss_array[min_index]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self._weighted_loss(y, self.predict(X))

    def _weighted_loss(self, y_true, y_pred):
        y_true = y_true.reshape(1, -1)
        loss = np.abs(y_true) @ (np.sign(y_true.T) != np.sign(y_pred))
        return loss
