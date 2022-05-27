import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train-
    # and test errors of AdaBoost in noiseless case
    a_boost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_errors = []
    test_errors = []
    for t in range(n_learners):
        train_errors.append(a_boost.partial_loss(train_X, train_y, t))
        test_errors.append(a_boost.partial_loss(test_X, test_y, t))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n_learners)), y=train_errors,
                             mode='lines',
                             name='train_error'))
    fig.add_trace(go.Scatter(x=list(range(n_learners)), y=test_errors,
                             mode='lines',
                             name='test_errors'))
    fig.update_layout(
        title={
            'text': "Model Error as Function of Number of Learners "})
    fig.write_image(f"/Users/tomerrozenstine/Documents/Uni/bcs/year2/Semester_B/67577_IML/Exercises /ex4/figures/loss_func_of_learners_{noise}.png")
    fig.show()

    # Question 2: Plotting decision surfaces
    symbols = np.array(["circle", "x"])
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[rf"$\textbf{{{t}}}$" for t in
                                        T],
                        horizontal_spacing=0.07, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(a_boost.partial_predict, lims[0], lims[1],
                                         showscale=False, T=t),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                   showlegend=False,
                                   marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)],
                                               colorscale=[custom[0],
                                                           custom[-1]],
                                               line=dict(color="black",
                                                         width=1)))],
                       rows=(i // 2) +1, cols=(i % 2) +1)

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title = {"text":"Decision Boundaries of Different #Learners"})
    fig.write_image(f"/Users/tomerrozenstine/Documents/Uni/bcs/year2/Semester_B/67577_IML/Exercises /ex4/figures/learenrs_sub_plots_{noise}.png")
    fig.show()
    # Question 3: Decision surface of best performing ensemble
    lowest_test_index = np.argmin(test_errors)
    lowest_test = test_errors[lowest_test_index]
    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=[f"lowest error ensemble size: {lowest_test_index + 1}, accuracy: {lowest_test}"],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    fig.add_traces([decision_surface(a_boost.partial_predict, lims[0], lims[1],
                                         showscale=False, T=lowest_test_index),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                   showlegend=False,
                                   marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)],
                                               colorscale=[custom[0],
                                                           custom[-1]],
                                               line=dict(color="black",
                                                         width=1)))])
    fig.write_image(f"/Users/tomerrozenstine/Documents/Uni/bcs/year2/Semester_B/67577_IML/Exercises /ex4/figures/lowest_error_model_{noise}.png")
    fig.show()
    # Question 4: Decision surface with weighted samples
    fig = make_subplots(rows=1, cols=1,
                        horizontal_spacing=0.01, vertical_spacing=.03)
    fig.add_traces([decision_surface(a_boost.partial_predict, lims[0], lims[1],
                                     showscale=False, T=n_learners),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                               showlegend=False,
                               marker=dict(color=train_y.astype(int),
                                           size=((a_boost.D_/np.max(a_boost.D_)) * 50),
                                           symbol=symbols[train_y.astype(int)],
                                           colorscale=[custom[0],
                                                       custom[-1]],
                                           line=dict(color="black",
                                                     width=1)))])
    fig.update_layout(
        title={
            'text': "Decision Surface With Weighted Markers Size"})
    fig.write_image(f"/Users/tomerrozenstine/Documents/Uni/bcs/year2/Semester_B/67577_IML/Exercises /ex4/figures/weighted_markers_{noise}.png")
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)

