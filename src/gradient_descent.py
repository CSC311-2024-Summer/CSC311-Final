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


def compute_gradients(alpha, beta, s, r, t, lambda_reg):
    n = len(t)
    grad_alpha = np.zeros_like(alpha)
    grad_beta = np.zeros_like(beta)

    for i in range(n):
        for j in range(n):
            if i != j:
                y_i = compute_y(alpha[i], beta[i], s[i], r[i])
                y_j = compute_y(alpha[j], beta[j], s[j], r[j])

                exp_term = np.exp(-(t[i] - t[j]) * (y_i - y_j))
                common_term = exp_term / (1 + exp_term)

                grad_alpha[i] += common_term * (-(t[i] - t[j])) * s[i]
                grad_alpha[j] += common_term * (-(t[i] - t[j])) * (-s[j])
                grad_beta[i] += common_term * (-(t[i] - t[j])) * r[i]
                grad_beta[j] += common_term * (-(t[i] - t[j])) * (-r[j])

    # Apply regularization
    # grad_alpha += lambda_reg * alpha
    grad_beta += lambda_reg * beta

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


def gradient_checking(alpha, beta, s, r, t, lambda_reg, epsilon=1e-5):
    func_alpha = lambda a: compute_loss(a, beta, s, r, t) + 0.5 * lambda_reg * np.sum(a ** 2)
    func_beta = lambda b: compute_loss(alpha, b, s, r, t) + 0.5 * lambda_reg * np.sum(b ** 2)

    grad_alpha, grad_beta = compute_gradients(alpha, beta, s, r, t, lambda_reg)
    num_grad_alpha = numerical_gradient(func_alpha, alpha, epsilon)
    num_grad_beta = numerical_gradient(func_beta, beta, epsilon)

    # Compare gradients
    diff_alpha = np.linalg.norm(grad_alpha - num_grad_alpha) / np.linalg.norm(grad_alpha + num_grad_alpha)
    diff_beta = np.linalg.norm(grad_beta - num_grad_beta) / np.linalg.norm(grad_beta + num_grad_beta)

    if diff_alpha > 0.001 or diff_beta > 0.001:
        print('Warning: Large gradient difference')
        raise ValueError
    print(f"Gradient Check - Alpha difference: {diff_alpha}")
    print(f"Gradient Check - Beta difference: {diff_beta}")

    return diff_alpha, diff_beta


def gradient_descent(alpha, beta, s, r, t, learning_rate, lambda_reg, grad_clip_threshold, num_iterations):
    for iteration in range(num_iterations):
        grad_alpha, grad_beta = compute_gradients(alpha, beta, s, r, t, lambda_reg)

        # Gradient clipping
        grad_alpha = np.clip(grad_alpha, -grad_clip_threshold, grad_clip_threshold)
        grad_beta = np.clip(grad_beta, -grad_clip_threshold, grad_clip_threshold)

        # Update alpha and beta
        alpha -= learning_rate * grad_alpha
        beta -= learning_rate * grad_beta

        # Compute and print the loss
        loss = compute_loss(alpha, beta, s, r, t)
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Loss = {loss}")
            gradient_checking(alpha, beta, s, r, t, lambda_reg)
        # print(f"Gradient alpha = {grad_alpha}, Gradient beta = {grad_beta}")

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
    s = generate_s_j(num_horses)  # please call get_s(d_bar), with d_bar from your output
    print(s)
    r = generate_r_j(num_horses)  # provide r (r = [r_1, ..., r_num_of_horses_in_race])
    print(r)
    t = generate_true_labels_ranking(num_horses)  # provide t (I assumed this would be the true rankings for 1 race)
    print(t)
    alpha = np.random.rand(num_horses)
    beta = np.random.rand(num_horses)

    # lambda reg = regularization parameter
    learning_rate = 0.01
    lambda_reg = 0.001
    grad_clip_threshold = 1.0
    num_iterations = 250

    # doing the gradient descent
    alpha, beta = gradient_descent(alpha, beta, s, r, t, learning_rate, lambda_reg, grad_clip_threshold,
                                   num_iterations)

    print("Final alpha:", alpha)
    print("Final beta:", beta)
