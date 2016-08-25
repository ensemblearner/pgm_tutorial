import numpy as np
from scipy.stats import multivariate_normal


def gen_data(mean, cov, num_points):
    data = np.random.multivariate_normal(mean, cov, num_points)
    return data


def e_step(data, means, covars, pis):
    responsibilities = np.zeros((means.shape[1], data.shape[0]) )
    i = 0
    for mean, covar, pi in zip(means, covars, pis):
        for j, point in enumerate(data):
            responsibilities[i, j] = pi * multivariate_normal.pdf(point, mean, covar)
        i+=1
    summation_res = np.sum(responsibilities, axis=0)
    return np.divide(responsibilities, summation_res)

def m_step(expectations, data):
    means = []
    covars = []
    pis = []
    N_k = np.sum(expectations, axis=1)

    for n_k, expectation in zip(N_k, expectations):
        expectation = expectation.reshape(-1,1)
        mu_new = np.divide(np.sum(data * expectation, axis=0), n_k)
        covar_new = np.zeros((data.shape[1], data.shape[1]))
        for exp,point in zip(expectation, data):
            diff = np.matrix(point-mu_new)
            covar_ =  np.multiply(exp[0], diff.T.dot(diff))
            covar_new = np.add(covar_new, covar_)

        covar_new = np.divide(covar_new, n_k)
        pi_k_new = n_k/data.shape[0]
        means.append(mu_new)
        covars.append(covar_new)
        pis.append(pi_k_new)
    return np.array(means), np.array(covars), pis

if __name__ == '__main__':
    mean = np.array([0, 0])
    cov = np.array([[1, 0.21], [0.11, 1]])
    data_1 = gen_data(mean, cov, 100)
    mean = np.array([4.5, 2.2])
    cov = np.array([[ 0.52, 0.26], [ 0.67, 0.93]])
    data_2 = gen_data(mean, cov, 50)
    data = np.concatenate((data_1, data_2))

    means = np.array([ [0.1, 0.2], [0.3, 0.1]])
    covars = [ np.array([[0.2, 0.1], [0.1, 0.3]]),
               np.array([[0.12, 0.31], [0.12, 0.33]])]
    pis = [0.3, 0.7]
    for i in range(200):
        expectation =  e_step(data, means, covars, pis)

        means, covars, pis = m_step(expectation, data)

    print "means"
    print means
    print "covars"
    print covars
    print "pis"
    print pis