import numpy as np


def compute_y(alpha, beta, s, r):
    return alpha * s + beta * r


def compute_loss(alpha, beta, s, r, t):
    n = len(t)
    loss = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                y_i = compute_y(alpha[i], beta[i], s[i], r[i])
                y_j = compute_y(alpha[j], beta[j], s[j], r[j])
                loss += np.log(1 + np.exp(-(t[i] - t[j]) * (y_i - y_j)))
    return loss


def compute_cost(alpha, beta, s, r, t, lambda_1, lambda_2):
    N = len(s)  # Number of races
    total_loss = 0.0
    n = len(t[0])  # Number of horses in each race

    for w in range(N):
        race_loss = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    y_i = compute_y(alpha[i], beta[i], s[w][i], r[w][i])
                    y_j = compute_y(alpha[j], beta[j], s[w][j], r[w][j])
                    diff = (t[w][i] - t[w][j]) * (y_i - y_j)
                    if diff > 0:
                        race_loss += np.log(1 + np.exp(-diff))
                    else:
                        race_loss += -diff + np.log(1 + np.exp(diff))
        total_loss += race_loss

    # cost = (total_loss / N) + (lambda_1 / n) * np.sum(np.abs(alpha)) + (lambda_2 / (2 * n)) * np.sum(beta ** 2)
    cost = (total_loss / N) + (lambda_1 / n) * np.sum(np.abs(alpha)) + (lambda_2 / n) * np.sum(np.abs(beta))
    return cost


def compute_gradients(alpha, beta, s, r, t, lambda_1, lambda_2):
    N = len(s)  # Number of races
    n = len(t[0])  # Number of horses in each race
    grad_alpha = np.zeros_like(alpha)
    grad_beta = np.zeros_like(beta)

    for w in range(N):
        for i in range(n):
            for j in range(n):
                if i != j:
                    y_i = compute_y(alpha[i], beta[i], s[w][i], r[w][i])
                    y_j = compute_y(alpha[j], beta[j], s[w][j], r[w][j])

                    diff = (t[w][i] - t[w][j]) * (y_i - y_j)
                    exp_term = np.exp(-diff)
                    common_term = exp_term / (1 + exp_term)

                    grad_alpha[i] += common_term * (-(t[w][i] - t[w][j])) * s[w][i]
                    grad_alpha[j] += common_term * (t[w][i] - t[w][j]) * s[w][j]
                    grad_beta[i] += common_term * (-(t[w][i] - t[w][j])) * r[w][i]
                    grad_beta[j] += common_term * (t[w][i] - t[w][j]) * r[w][j]

    grad_alpha /= N
    grad_beta /= N

    # Add regularization gradient
    grad_alpha += (lambda_1 / n) * np.sign(alpha)
    # grad_beta += (lambda_2 / n) * beta
    grad_beta += (lambda_2 / n) * np.sign(beta)

    return grad_alpha, grad_beta


def numerical_gradient(func, param, epsilon=1e-5):
    num_grad = np.zeros_like(param)
    perturb = np.zeros_like(param)

    for i in range(len(param)):
        perturb[i] = epsilon
        loss_plus = func(param + perturb)
        loss_minus = func(param - perturb)
        num_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
        perturb[i] = 0  # Reset perturbation

    return num_grad


def gradient_checking(alpha, beta, s, r, t, lambda_1, lambda_2, epsilon=1e-5):
    func_alpha = lambda a: compute_cost(a, beta, s, r, t, lambda_1, lambda_2)
    func_beta = lambda b: compute_cost(alpha, b, s, r, t, lambda_1, lambda_2)

    grad_alpha, grad_beta = compute_gradients(alpha, beta, s, r, t, lambda_1, lambda_2)
    num_grad_alpha = numerical_gradient(func_alpha, alpha)
    num_grad_beta = numerical_gradient(func_beta, beta)

    alpha_diff = np.linalg.norm(grad_alpha - num_grad_alpha)
    beta_diff = np.linalg.norm(grad_beta - num_grad_beta)

    print(f"Gradient Check - Alpha difference: {alpha_diff}")
    print(f"Gradient Check - Beta difference: {beta_diff}")

    return alpha_diff, beta_diff


def gradient_descent(alpha, beta, s, r, t, learning_rate, lambda_1, lambda_2, grad_clip_threshold, num_iterations,
                     convergence_threshold=1e-2):
    prev_cost = float('inf')

    for iteration in range(num_iterations):
        grad_alpha, grad_beta = compute_gradients(alpha, beta, s, r, t, lambda_1, lambda_2)

        # Apply gradient clipping
        grad_alpha = np.clip(grad_alpha, -grad_clip_threshold, grad_clip_threshold)
        grad_beta = np.clip(grad_beta, -grad_clip_threshold, grad_clip_threshold)

        # Update alpha and beta
        alpha -= learning_rate * grad_alpha
        beta -= learning_rate * grad_beta

        cost = compute_cost(alpha, beta, s, r, t, lambda_1, lambda_2)

        # Check for convergence
        if abs(prev_cost - cost) < convergence_threshold:
            print(f"converged it: {iteration}: Cost = {cost}")
            break

        prev_cost = cost

        if iteration % 1000 == 0:
            pass
            # pass
            # print(f"Iteration {iteration}: Cost = {cost}")
            # gradient_checking(alpha, beta, s, r, t, lambda_1, lambda_2)

    return alpha, beta
def get_s(d_bar):
    return np.exp(-d_bar)


def generate_s_j(num_horses):
    d_bar = np.random.normal(num_horses / 2, num_horses / 4 - 0.5, size=num_horses)
    print(d_bar)

    # Apply exponential transformation
    S_j = np.exp(-d_bar)

    return S_j


def generate_r_j(num_horses, mean=12, std=7):
    r_j = np.random.normal(mean, std, size=num_horses)
    M_r = np.mean(r_j)
    sigma_r = np.std(r_j)
    r_j_normalized = (r_j - M_r) / sigma_r

    return r_j_normalized


def generate_true_labels_ranking(num_horses):
    # True labels are a permutation of {1, 2, ..., num_horses}
    true_ranks = np.random.permutation(np.arange(1, num_horses + 1))
    return true_ranks


def generate_true_labels_regression(num_horses, mean=12, std=7):
    # True labels are finish times in seconds, normally distributed
    true_times = np.random.normal(mean, std, size=num_horses)
    return true_times


if __name__ == '__main__':
    num_horses = 8
    num_races = 10
    s = [generate_s_j(num_horses) for _ in range(num_races)]
    r = [generate_r_j(num_horses) for _ in range(num_races)]
    t = [generate_true_labels_ranking(num_horses) for _ in range(num_races)]

    alpha = np.random.rand(num_horses)
    beta = np.random.rand(num_horses)

    learning_rate = 0.01
    lambda_1 = 0.001
    lambda_2 = 0.001
    grad_clip_threshold = 1.0
    num_iterations = 1000

    alpha, beta = gradient_descent(alpha, beta, s, r, t, learning_rate, lambda_1, lambda_2, grad_clip_threshold,
                                   num_iterations)

    print("Final alpha:", alpha)
    print("Final beta:", beta)
