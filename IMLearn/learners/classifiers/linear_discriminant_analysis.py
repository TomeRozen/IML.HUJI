from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from scipy.stats import multivariate_normal


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features' covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        self.pi_ = counts / sum(counts)

        self.mu_ = np.zeros((len(self.classes_), X.shape[1]))
        label_index_dict = {}
        # map label to its index:
        for i, k in enumerate(self.classes_):
            label_index_dict[k] = i
        # sum label's samples:
        for index, label in enumerate(y):
            self.mu_[label_index_dict[label]] += X[index]
        # divide by number of samples of each class:
        self.mu_ /= counts.reshape(-1, 1)

        # calculating self.cov:
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        for index, label in enumerate(y):
            error = np.array(X[index] - self.mu_[label_index_dict[label]])
            self.cov_ += np.outer(error, error)
        self.cov_ /= (X.shape[0] - len(self.classes_))
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        ak_matrix = self._cov_inv @ self.mu_.transpose()  # num_features X num_classes
        bk_vec = np.log(self.pi_) - (0.5 * (np.diag(self.mu_ @ self._cov_inv @
                                                    self.mu_.transpose())))  # num_classes
        classes_indexes = ((X @ ak_matrix) + bk_vec).argmax(1)
        classes_indexes = self.classes_[classes_indexes]
        prediction = np.zeros((X.shape[0],))
        for index, row in enumerate(classes_indexes):
            prediction[index] = self.classes_[row]
        return prediction

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihood = np.zeros((X.shape[0], len(self.classes_)))
        for index, row in enumerate(self.mu_):
            likelihood[:, index] = multivariate_normal.pdf(X, mean=row, cov=self.cov_)*self.pi_[index]

        return likelihood

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
