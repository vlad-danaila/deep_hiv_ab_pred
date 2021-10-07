import torch as t
from util.tools import normalize, unnormalize, to_torch, to_numpy
from typing import Tuple, Union
from scipy.stats import norm
import numpy as np
import math

'''
Reproduced from:
https://github.com/vlad-danaila/deep_learning_censored_regression/blob/master/deep_tobit/normal_cumulative_distribution_function.py
'''
class __CDF(t.autograd.Function):

    @staticmethod
    def forward(ctx, x: t.Tensor) -> t.Tensor:
        type, device = x.dtype, x.device
        _x = to_numpy(x)
        pdf = to_torch(norm.pdf(_x), type = type, device = device, grad = False)
        ctx.save_for_backward(pdf)
        return to_torch(norm.cdf(_x), type = type, device = device, grad = False)

    @staticmethod
    def backward(ctx, grad_output):
        pdf = ctx.saved_tensors[0]
        grad = None
        if ctx.needs_input_grad[0]:
            grad = grad_output * pdf
        return grad

cdf = __CDF.apply

'''
Reproduced from:
https://github.com/vlad-danaila/deep_learning_censored_regression/blob/master/deep_tobit/loss.py
'''
class Reparametrized_Scaled_Tobit_Loss(t.nn.Module):

    def __init__(self, gamma: t.Tensor, device: Union[t.device, str, None], truncated_low: float = None, truncated_high: float = None, epsilon: float = 1e-40):
        super(Reparametrized_Scaled_Tobit_Loss, self).__init__()
        self.gamma = gamma
        self.device = device
        self.truncated_low = truncated_low
        self.truncated_high = truncated_high
        self.epsilon = t.tensor(epsilon, dtype=t.float32, device=device, requires_grad=False)

    def forward(self, x: Tuple[t.Tensor, t.Tensor, t.Tensor], y: Tuple[t.Tensor, t.Tensor, t.Tensor]) -> t.Tensor:
        x_single_value, x_left_censored, x_right_censored = x
        y_single_value, y_left_censored, y_right_censored = y
        N = len(y_single_value) + len(y_left_censored) + len(y_right_censored)

        gamma = t.abs(self.gamma)

        # Step 1: compute loss for uncensored data based on pdf:
        # -sum(ln(gamma) + ln(pdf(gamma * y - x)))
        log_likelihood_pdf = to_torch(0, device = self.device, grad = True)
        if len(y_single_value) > 0:
            log_likelihood_pdf = -t.sum(t.log(gamma + self.epsilon) - ((gamma * y_single_value - x_single_value) ** 2) / 2)

        # Step 2: compute loss for left censored data:
        # -sum(ln(cdf(gamma * y - x) - cdf(gamma * truncation - x)))
        log_likelihood_cdf = to_torch(0, device = self.device, grad = True)
        if len(y_left_censored) > 0:
            truncation_low_penalty = 0 if not self.truncated_low else cdf(gamma * self.truncated_low - x_left_censored)
            log_likelihood_cdf = -t.sum(t.log(cdf(gamma * y_left_censored - x_left_censored) - truncation_low_penalty + self.epsilon))

        # Step 3: compute the loss for right censored data:
        # -sum(ln(cdf(x - gamma * y) - cdf(x - gamma * truncation)))
        # Notice that: log(1 - cdf(z)) = log(cdf(-z)), thus compared to step 2, the signs for gamma and x are swapped
        log_likelihood_1_minus_cdf = to_torch(0, device = self.device, grad = True)
        if len(y_right_censored) > 0:
            truncation_high_penalty = 0 if not self.truncated_high else cdf(-gamma * self.truncated_high + x_right_censored)
            log_likelihood_1_minus_cdf = -t.sum(t.log(cdf(-gamma * y_right_censored + x_right_censored) - truncation_high_penalty + self.epsilon))

        log_likelihood = log_likelihood_pdf + log_likelihood_cdf + log_likelihood_1_minus_cdf

        return log_likelihood

    def get_scale(self) -> t.Tensor:
        return 1 / t.abs(self.gamma)

def estimate_censored_mean(single_valued = [], left_censored = [], right_censored = [],
                           lr = 1e-3, max_iterations = 10_000, early_stop_patience = 5, epsilon = 1e-6, output_std = False):
    all = np.array(single_valued + left_censored + right_censored)
    data_mean, data_std = all.mean(), all.std() + 1e-10

    single_valued = t.tensor(single_valued, dtype=t.float32)
    left_censored = t.tensor(left_censored, dtype=t.float32)
    right_censored = t.tensor(right_censored, dtype=t.float32)

    single_valued = normalize(single_valued, data_mean, data_std)
    left_censored = normalize(left_censored, data_mean, data_std)
    right_censored = normalize(right_censored, data_mean, data_std)
    zero_normalized = normalize(0, data_mean, data_std)

    delta = to_torch(0, grad=True)
    # tuple for single valued, left censored, right censored
    x_tuple = (delta, delta, delta)
    y_tuple = (single_valued, left_censored, right_censored)
    gamma = to_torch(1, device = 'cpu', grad = True)
    tobit = Reparametrized_Scaled_Tobit_Loss(gamma, device = 'cpu', truncated_low = zero_normalized)
    optimizer = t.optim.SGD([delta, gamma], lr = lr)

    patience = early_stop_patience
    for i in range(max_iterations):
        prev_delta, prev_gamma = delta.clone(), gamma.clone()
        optimizer.zero_grad()
        loss = tobit(x_tuple, y_tuple)
        loss.backward()
        optimizer.step()
        early_stop = math.fabs(delta - prev_delta) + math.fabs(gamma - prev_gamma) < epsilon
        if early_stop:
            patience -= 1
            if patience == 0:
                break
        else:
            patience = early_stop_patience

    mean, std = delta / gamma, 1 / gamma
    mean, std = unnormalize(mean, data_mean, data_std), std * data_std
    return (mean.item(), std.item()) if output_std else mean.item()