import numpy as np


def pairwise_loss(t_i, t_j, y_i, y_j):
    return np.log(1 + np.exp(-(t_i - t_j) * (y_i - y_j)))


def grad_pairwise_alpha(t_i, t_j, y_i, y_j, s_i, s_j):
    """
    gradient of pairwise loss with respect to alpha
    returns both gradients, w.r.t alpha_i and alpha_j respectively
    """
    exp_ij = np.exp(-(t_i - t_j) * (y_i - y_j))
    return ((1 / (1 + exp_ij)) * (-(t_i - t_j)) * s_i).sum(), ((1 / (1 + exp_ij)) * (t_i - t_j) * s_j).sum()


def grad_pairwise_beta(t_i, t_j, y_i, y_j, r_i, r_j):
    """
    gradient of pairwise loss with respect to beta
    returns both gradients, w.r.t beta_i and beta_j respectively
    """
    exp_term = np.exp(-(t_i - t_j) * (y_i - y_j))
    return ((1 / (1 + exp_term)) * (-(t_i - t_j)) * r_i).sum(), ((1 / (1 + exp_term)) * (t_i - t_j) * r_j).sum()


def compute_grad_pairwise(alpha, beta, S, R, T):
    """
    :param alpha:
    :param beta:
    :param S:
    :param R:
    :param T: the N by 1 column vector of the true labels
    :return:
    """
    n = T.shape[0]
    dL_dalpha = np.zeros_like(alpha)
    dL_dbeta = np.zeros_like(beta)

    for i in range(n):
        for j in range(n):
            if i != j:
                y_i = alpha[i] * S[i] + beta[i] * R[i]
                y_j = alpha[j] * S[j] + beta[j] * R[j]

                exp_term = np.exp(-(T[i] - T[j]) * (y_i - y_j))
                common_factor = 1 / (1 + exp_term) * (-(T[i] - T[j]))

                dL_dalpha[i] += common_factor * S[i]
                dL_dalpha[j] -= common_factor * S[j]

                dL_dbeta[i] += common_factor * R[i]
                dL_dbeta[j] -= common_factor * R[j]

    return dL_dalpha, dL_dbeta


def total_loss(alpha, beta, S, R, T):
    n = T.shape[0]
    total_loss = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                y_i = alpha[i] * S[i] + beta[i] * R[i]
                y_j = alpha[j] * S[j] + beta[j] * R[j]
                total_loss += pairwise_loss(T[i], T[j], y_i, y_j)
    return total_loss


def gradient_descent(alpha, beta, S, R, T, learning_rate, iterations):
    for _ in range(iterations):
        dL_dalpha, dL_dbeta = compute_grad_pairwise(alpha, beta, S, R, T)

        # Update alpha and beta
        alpha -= learning_rate * dL_dalpha
        beta -= learning_rate * dL_dbeta

        print(f"Iteration {_}: Loss = {total_loss(alpha, beta, S, R, T)}")

    return alpha, beta


# Function to generate S from exp(f(x)) where f(x) is between 1 and 8
def generate_S(n):
    f_x = np.random.uniform(1, 8, size=n)
    S = np.exp(f_x)
    return S


# Function to generate R as the output of a regression model
def generate_R(n):
    # Simulate a regression model output
    X = np.random.rand(n, 1)
    true_coefficient = 3.5
    noise = np.random.randn(n, 1) * 0.5
    R = true_coefficient * X + noise
    return R.flatten()


# Generate T as binary labels
def generate_T(n):
    return np.random.randint(0, 2, size=n)


if __name__ == '__main__':
    # test
    n = 100
    S = generate_S(n)
    R = generate_R(n)
    T = generate_T(n)
    print("S:", S)
    print("R:", R)
    print("T:", T)

    alpha = np.random.rand(n)
    beta = np.random.rand(n)

    learning_rate = 0.01
    iterations = 1000

    alpha, beta = gradient_descent(alpha, beta, S, R, T, learning_rate, iterations)

    print("Alpha:", alpha)
    print("Beta:", beta)
