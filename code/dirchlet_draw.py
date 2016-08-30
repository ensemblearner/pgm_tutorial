import matplotlib.pyplot as plt
import numpy as np


def draw_dirichlet(alphas, num_trials):
    samples = np.random.dirichlet(alphas, num_trials).transpose()
    for i, alpha in enumerate(alphas):
        plt.barh(range(num_trials), samples[i], left=sum(samples[:i]), color=np.random.rand(3, 1))
    plt.title('draws from dirichlet')
    plt.show()

draw_dirichlet(alphas=[10, 20, 17], num_trials=50)
