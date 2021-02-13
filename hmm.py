import numpy as np
from hmmlearn.hmm import GaussianHMM
from midi import MIDI, Encoded
from prepare_data import process, get_notes_range, restore
import pickle
import cv2
import os
from lstm_settings import base_settings
import matplotlib.pyplot as plt


# ==================================================================================================================================
# Source of following 2 functions:
# https://stats.stackexchange.com/questions/384556/how-to-infer-the-number-of-states-in-a-hidden-markov-model-with-gaussian-mixture
# (first answer) by Sashi
# ==================================================================================================================================

def bic_general(likelihood_fn, k, data):
    """likelihood_fn: Function. Should take as input X and give out   the log likelihood
                  of the data under the fitted model.
           k - int. Number of parameters in the model. The parameter that we are trying to optimize.
                    For HMM it is number of states.
                    For GMM the number of components.
           X - array. Data that been fitted upon.
    """
    try:
        likelihood = likelihood_fn(data)
        bic = np.log(len(data)) * k - 2 * likelihood
    except ValueError:
        bic = np.infty
    return bic


def bic_hmmlearn(data, max_n_components=10):
    lowest_bic = np.infty
    bic = []
    range_start = 1
    n_states_range = range(range_start, max_n_components)
    best_hmm = None

    for n_components in n_states_range:
        hmm_curr = GaussianHMM(n_components=n_components, covariance_type='diag')
        hmm_curr.fit(data)

        # Calculate number of free parameters
        # free_parameters = for_means + for_covars + for_transmat + for_startprob
        # for_means & for_covars = n_features*n_components
        n_features = hmm_curr.n_features
        free_parameters = 2 * (n_components * n_features) + n_components * (n_components - 1) + (n_components - 1)

        bic_curr = bic_general(hmm_curr.score, free_parameters, data)
        bic.append(bic_curr)
        if bic_curr < lowest_bic:
            lowest_bic = bic_curr
            print(f"Found better n_components {n_components}")
        best_hmm = hmm_curr

        if n_components > 1:
            plt.plot(range(range_start, n_components + 1), bic)
            plt.title('BIC over number of components')
            plt.ylabel('Bayesian information criterion')
            plt.xlabel('N components')
            plt.show()

    return best_hmm, bic, n_states_range


# ==================================================================================================================================
# End of stackoverflow code
# ==================================================================================================================================

def create_model(input_data, n_components, cov_type, input_name):
    hmm = GaussianHMM(n_components=n_components, covariance_type=cov_type)
    hmm.fit(input_data)
    with open(f"output/hmm_{input_name}_{n_components}_{cov_type}.pkl", "wb") as file:
        pickle.dump(hmm, file)
        print("Exported model to file!")
    return hmm


def load_model(n_components, cov_type, input_name):
    with open(f"output/hmm_{input_name}_{n_components}_{cov_type}.pkl", "rb") as file:
        print("Imported model from file!")
        return pickle.load(file)


def get_model(input_data, n_components, cov_type, input_name, recreate_override=False):
    if recreate_override or not os.path.isfile(f"output/hmm_{input_name}_{n_components}_{cov_type}.pkl"):
        print(f"Creating model (n_components={n_components}, input_name={input_name}, cov_type={cov_type})")
        return create_model(input_data, n_components, cov_type, input_name)
    print(f"Loading model (n_components={n_components}, input_name={input_name}, cov_type={cov_type})")
    return load_model(n_components, cov_type, input_name)


def main():
    m = MIDI(base_settings.ticks_per_second)
    input_file = os.path.abspath('input/unfin.midi')
    input_name = os.path.splitext(os.path.basename(input_file))[0]

    n_components = 36
    cov_type = 'diag'
    samples_threshold = 0.97

    encoded = m.from_midi(input_file)
    data = encoded.data.T
    start, end = get_notes_range(data=data)
    input_data = process(data, start, end, add_end_token=False)

    input_data[input_data > 0] = 1
    cv2.imwrite(f"output/hmm_input_{input_name}.png", input_data.T * 255)
    # input_data = input_data[:, input_data.sum(axis=0) > 0]

    # best_hmm, bic, n_components_range = bic_hmmlearn(input_data, 70)
    # model = best_hmm
    # plt.plot(n_components_range, bic)
    # plt.title('BIC over number of components')
    # plt.ylabel('Bayesian information criterion')
    # plt.xlabel('N components')
    # plt.savefig(f"output/hmm_{input_name}_bic.png")
    # plt.show()

    model = get_model(input_data, n_components, cov_type, input_name, False)

    samples = model.sample(500)[0]
    samples[samples < samples_threshold] = 0
    samples[samples > 0] = 100
    cv2.imwrite(f"output/hmm_samples_{input_name}_{n_components}_{cov_type}.png", samples.T * 2)

    print(samples)
    samples_data = samples.clip(0, 127).astype(np.int8)
    restored_data = restore(samples_data, start, end, remove_end_token=False)
    m.to_midi(Encoded(restored_data.T, *encoded[1:]), f"output/hmm_{input_name}_{n_components}_{cov_type}.midi")


if __name__ == '__main__':
    main()
