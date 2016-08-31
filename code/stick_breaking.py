import numpy as np
import collections
import matplotlib.pyplot as plt

class StickBreaking(object):
    def __init__(self, alpha, k):
        self.alpha = alpha
        self.k = k

    @property
    def pis(self):
        #stick breaking procedure
        betas = np.random.beta(1, self.alpha, self.k)
        # compute prod(1- beta)_i^n-1
        cum_prods = np.append(1, np.cumproduct(1 - betas[:-1]))
        prods = betas * cum_prods
        return prods / prods.sum()

    @property
    def H(self):
        thetas = np.random.normal(0, 1, self.k)
        return thetas

    def generate_G(self):
        #G = sum(pi*delta)
        xs = np.random.multinomial(self.k, self.pis)
        dp_clusters = self.H[xs]
        counts = collections.Counter(dp_clusters).values()
        plt.bar(set(dp_clusters), counts, width=0.01)
        plt.title("dp via stick breaking")
        plt.xlabel("atom locations")
        plt.ylabel("atoms collided")
        plt.show()


if __name__ == '__main__':

    sb = StickBreaking(alpha=20, k=100)
    sb.generate_G()