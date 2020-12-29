import numpy as np
from hmmlearn.hmm import GaussianHMM
from midi import from_midi, to_midi, Encoded
import pickle
import cv2


def fit_model(data, n_components):
    hmm = GaussianHMM(n_components=n_components, covariance_type='full', n_iter=100)
    hmm.fit(data)
    return hmm


def best_model(data):
    # 17, 3100
    best_score = float('-inf')
    best_param = (None, None)
    scores = []

    for n_components in np.arange(1, 40, 1):
        # for n_iter in np.arange(100, 200, 500):
        print(f"Trying {n_components}")
        fit_iters = 3
        avg_score = 0
        hmm = None
        for i in range(fit_iters):
            hmm = fit_model(data, n_components)
            avg_score += hmm.score(data)
        # todo try bayesian information criterion for selecting best model to take into account the complexity
        avg_score /= fit_iters
        scores.append((n_components, avg_score))
        if avg_score > best_score:
            print(f"Found new better model at n_components: {n_components}, score: {avg_score}")
            best_score = avg_score
            best_param = n_components, hmm

    print(best_score, best_param)
    _, hmm = best_param
    return hmm, scores


def create_model(input_data, n_components):
    hmm = fit_model(input_data, n_components)
    with open(f"data/hmm_{n_components}.pkl", "wb") as file:
        pickle.dump(hmm, file)
        print("Exported model to file!")
    return hmm


def load_model(n_components):
    with open(f"data/hmm_{n_components}.pkl", "rb") as file:
        print("Imported model from file!")
        return pickle.load(file)


def main():
    n_components = 45
    samples_threshold = 0.95

    encoded = from_midi('data/unfin.midi')
    input_data = encoded.data.T
    input_data[input_data > 0] = 1
    cv2.imwrite("data/hmm_input.png", input_data.T * 255)
    # input_data = input_data[:, input_data.sum(axis=0) > 0]

    model = create_model(input_data, n_components)
    # model = load_model(n_components)

    samples = model.sample(500)[0]
    samples[samples < samples_threshold] = 0
    samples = samples * 127
    cv2.imwrite("data/hmm_samples.png", samples.T * 2)

    print(samples)
    samples_data = np.rint(samples.T).astype(int).clip(0, 127)
    sample_encoded = Encoded(samples_data, encoded.key_signature, encoded.time_signature, encoded.bpm)
    to_midi(sample_encoded, "data/predicted.midi")


if __name__ == '__main__':
    main()
