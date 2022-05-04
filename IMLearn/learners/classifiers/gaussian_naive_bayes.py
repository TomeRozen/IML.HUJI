from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from scipy.stats import multivariate_normal


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)

        num_of_classes = len(self.classes_)
        num_of_features = X.shape[1]

        self.pi_ = counts / sum(counts)
        self.mu_ = np.zeros((num_of_classes, num_of_features))
        label_index_dict = {}
        # map label to its index:
        for i, k in enumerate(self.classes_):
            label_index_dict[k] = i
        # sum label's samples:
        for index, label in enumerate(y):
            self.mu_[label_index_dict[label]] += X[index]
        # divide by number of samples of each class:
        self.mu_ /= counts.reshape(-1, 1)

        self.vars_ = np.zeros((num_of_classes, num_of_features))
        for index, label in enumerate(y):
            self.vars_[label_index_dict[label]] += (X[index] - self.mu_[label_index_dict[label]])**2
        # divide by number of samples of each class:
        self.vars_ /= counts.reshape(-1, 1)

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
        return self.classes_[self.likelihood(X).argmax(1)]

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
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        likelihood = np.zeros((X.shape[0], len(self.classes_)))
        for index, row in enumerate(self.mu_):
            mean = row
            cov = np.diag(self.vars_[index])
            likelihood[:, index] = multivariate_normal.pdf(X, mean=mean, cov=cov) \
                                   * self.pi_[index]

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

