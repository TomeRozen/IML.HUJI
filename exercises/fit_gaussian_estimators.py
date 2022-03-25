from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    normal_arr = np.random.normal(10, 1, 1000)
    uni_var = UnivariateGaussian().fit(normal_arr)
    print("(", uni_var.mu_, ",", uni_var.var_, ")")

    # Question 2 - Empirically showing sample mean is consistent
    estimated_mean = []
    sample_size_arr = np.array(range(10, 1010, 10))
    for sz in sample_size_arr:
        X = normal_arr[:sz]
        estimated_mean.append(np.mean(X))

    go.Figure([go.Scatter(x=sample_size_arr,
                          y=estimated_mean,
                          mode='markers+lines',
                          name=r'$\widehat\mu$'),
               go.Scatter(x=sample_size_arr,
                          y=[uni_var.mu_] * len(sample_size_arr),
                          mode='lines',
                          name=r'$\mu$')],
              layout=go.Layout(
                  title=r"$\text{Estimation of Expectation As Function Of "
                        r"Number Of Samples From the Fitted Array}$",
                  xaxis_title="$\\text{number of samples from the fitted"
                              " array}$",
                  yaxis_title="r$\hat\mu$",
                  height=800)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure([go.Scatter(x=normal_arr,
                          y=uni_var.pdf(normal_arr),
                          mode='markers')],
              layout=go.Layout(
                  title=r"$\text{Samples Scatter as a Function of Their PDF}$",
                  xaxis_title="$\\text{sample value}$",
                  yaxis_title="$\\text{sample pdf}$",
                  height=800)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    mult_var_noraml_array = np.random.multivariate_normal(mu, sigma, 1000)
    mult_var_model = MultivariateGaussian().fit(mult_var_noraml_array)
    print("Fitted Model's mu Is: ")
    print(mult_var_model.mu_)
    print()
    print("Fitted Model's Covariance Matrix Is: ")
    print(mult_var_model.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    log_like_matrix = np.zeros((len(f1), len(f3)))
    for i, value_1 in enumerate(f1):
        for j, value_3 in enumerate(f3):
            f_mu = np.array([value_1, 0, value_3, 0])
            log_like_matrix[i, j] = MultivariateGaussian()\
                .log_likelihood(f_mu, sigma, mult_var_noraml_array)

    fig = px.imshow(log_like_matrix,
                    labels=dict(x="f3 Value", y="f1 Value"),
                    x=f3,
                    y=f1)
    fig.update_layout(title='Log-Likelihood of Samples Drawn, Given f1 and f3 Values of Expectancy')
    # fig.show()

    # Question 6 - Maximum likelihood
    argmax = log_like_matrix.argmax()
    argmax_i = argmax // len(f1)
    argmax_j = argmax % len(f3)
    print("argmax values:")
    print("f1 value: ", np.round(f1[argmax_i],4), ", f3 value: ", np.round(f3[argmax_j], 4))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
    arr = np.array(
        [1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3,
         1, -4, 1, 2, 1,
         -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1,
         0, 3, 5, 0, -2])
    print(UnivariateGaussian().log_likelihood(1,1,arr))
    print(UnivariateGaussian().log_likelihood(10,1,arr))