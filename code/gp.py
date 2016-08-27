import numpy as np
class GPReg(object):

    def __init(self, X, y, kfunc):
        self.X = X
        self.y = y
        self.m, self.n = X.shape
        self.kfunc = kfunc
        self.K = self.compute_K()

    def compute_K(self):
        kernel = np.zeros((self.m, self.m))
        for i, row in enumerate(self.X):
            for j, ele in enumerate(row):
                for k, ele_2 in enumerate(row):
                    kernel[j, k] = self.kfunc(ele, ele_2)
        return kernel


    def mean_post_infer(self, K_star, y):
        return K_star.dot(np.linalg.inv(self.K)).dot(y)

    def covar_post_infer(self, K_star_star, K_star, K):
        return K_star_star - K_star.dot(np.linalg.inv(self.K)).K_star.T


