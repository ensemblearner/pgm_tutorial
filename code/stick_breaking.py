import numpy as np

def stick_breaking(alpha, k):
    betas = np.random.beta(1, alpha, k)
    # compute prod(1- beta)_i^n-1
    cum_prods = np.append(1, np.cumproduct(1-betas[:-1]))
    prods = betas * cum_prods
    return prods/prods.sum()

print stick_breaking(2, 20)