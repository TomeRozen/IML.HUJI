import os.path

import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(os.path.join("../datasets", f))

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def evaluation_callback(perceptron, sample, answer):
            losses.append(perceptron._loss(X, y))

        Perceptron(include_intercept=False, callback=evaluation_callback).fit(X, y)
        # Plot figure of loss as function of fitting iteration
        line_plt = px.line(x=range(1, len(losses) + 1), y=losses,
                           title="Perceptron Loss per Iteration of: " + n)
        line_plt.update_yaxes(title_text="Perceptron Loss")
        line_plt.update_xaxes(title_text="Number of Iteration")
        # line_plt.write_image("/Users/tomerrozenstine/Documents/Uni/bcs/year2/"
        #                      "Semester_B/67577_IML/Exercises /ex3/figures/"
        #                      f"{n}.png")
        line_plt.show()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    symbols = np.array(["circle", "triangle-up", "hourglass"])
    colors = np.array(["red", "blue", "cyan"])
    # symbols = SymbolValidator().values
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(os.path.join("../datasets", f))
        # Fit models and predict over training set
        lda, gaussian_nb = LDA(), GaussianNaiveBayes()
        lda.fit(X, y)
        gaussian_nb.fit(X, y)

        gnb_predict = gaussian_nb.predict(X)
        lda_predict = lda.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        gnb_accuracy = accuracy(y, gnb_predict)
        lda_accuracy = accuracy(y, lda_predict)
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            f"Gaussian Naive Bayes Predictions, accuracy is: {gnb_accuracy}"
            , f"LDA predictions, accuracy is: {lda_accuracy}"],
                            vertical_spacing=0.10)

        # Add traces for data-points setting symbols and colors

        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                       marker=dict(
                           color=colors[gaussian_nb.predict(X).astype(int)],
                           symbol=symbols[y])), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                       marker=dict(color=colors[lda.predict(X).astype(int)],
                                   symbol=symbols[y])), row=1, col=2)
        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(
            go.Scatter(x=gaussian_nb.mu_[:, 0], y=gaussian_nb.mu_[:, 1],
                       mode="markers",
                       marker=dict(
                           color="Black",
                           symbol="x")), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers",
                       marker=dict(color="Black",
                                   symbol="x")), row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for index, label in enumerate(lda.classes_):
            fig.add_trace(get_ellipse(gaussian_nb.mu_[index], np.diag(gaussian_nb.vars_[index])), row=1, col=1)
            fig.add_trace(get_ellipse(lda.mu_[index], lda.cov_), row=1, col=2)

        fig.update_layout(height=600, width=1200, title_text=f+"\n", title_x=0.5, title_font_size=25, showlegend=False)
        fig.write_image("/Users/tomerrozenstine/Documents/Uni/bcs/year2/"
                             "Semester_B/67577_IML/Exercises /ex3/figures/"
                             f"{f}.png")


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
