import datetime
import warnings
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from hmmlearn.hmm import GaussianHMM
import pandas as pd


def fitHMM(input_data, n_samples):
    # fit Gaussian HMM to Q
    model = GaussianHMM(n_components=2, n_iter=1000).fit(input_data)

    # classify each observation as state 0 or 1
    hidden_states = model.predict(input_data)

    # find parameters of Gaussian HMM
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]), np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)

    # find log-likelihood of Gaussian HMM
    logProb = model.score(input_data)

    # generate nSamples from Gaussian HMM
    samples = model.sample(n_samples)

    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states

    return hidden_states, mus, sigmas, P, logProb, samples


def main():
    input_data = np.loadtxt("F.txt")
    logQ = np.log(input_data)
    hidden_states, mus, sigmas, P, logProb, samples = fitHMM(logQ, 100)

    # todo run with multiple initializations and take best model (score())
    # also add years to input
    plt.figure()
    plt.title("Data")
    plt.plot(input_data)
    plt.show()

    hmm = GaussianHMM(n_components=5, n_iter=1000)
    hmm.fit(input_data)
    print(f"Converged? {hmm.monitor_.converged}")
    samples = hmm.sample(200)
    plt.figure()
    plt.title("Samples")
    plt.plot(samples[0])
    plt.show()
    print(samples)
    #
    # num_samples = 300
    # samples, _ = hmm.sample(num_samples)
    #
    # plt.figure()
    # plt.title('Difference percentages')
    # plt.plot(np.arange(num_samples), samples[:, 0], c='black')
    #
    # plt.figure()
    # plt.title('Volume of shares')
    # plt.plot(np.arange(num_samples), samples[:, 1], c='black')
    # plt.ylim(ymin=0)
    # plt.show()


if __name__ == '__main__':
    main()
