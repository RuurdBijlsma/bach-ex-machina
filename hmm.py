import numpy as np
from hmmlearn.hmm import GaussianHMM
from midi import MIDI, Encoded
from prepare_data import process, get_notes_range, restore
import pickle
import cv2
import os


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


def create_model(input_data, n_components, input_name):
    hmm = fit_model(input_data, n_components)
    with open(f"data/hmm_{input_name}_{n_components}.pkl", "wb") as file:
        pickle.dump(hmm, file)
        print("Exported model to file!")
    return hmm


def load_model(n_components, input_name):
    with open(f"data/hmm_{input_name}_{n_components}.pkl", "rb") as file:
        print("Imported model from file!")
        return pickle.load(file)


def get_model(input_data, n_components, input_name, recreate_override=False):
    if recreate_override or not os.path.isfile(f"data/hmm_{input_name}_{n_components}.pkl"):
        print(f"Creating model (n_components={n_components}, input_name={input_name})")
        return create_model(input_data, n_components, input_name)
    print(f"Loading model (n_components={n_components}, input_name={input_name})")
    return load_model(n_components, input_name)


def main():
    m = MIDI()
    input_file = os.path.abspath('data/unfin.midi')
    input_name = os.path.splitext(os.path.basename(input_file))[0]

    n_components = 65
    samples_threshold = 0.97

    encoded = m.from_midi(input_file)
    start, end = get_notes_range("bach")
    encoded = process(encoded, start, end)

    input_data = encoded.data.T
    input_data[input_data > 0] = 1
    cv2.imwrite(f"data/hmm_input_{input_name}.png", input_data.T * 255)
    # input_data = input_data[:, input_data.sum(axis=0) > 0]

    model = get_model(input_data, n_components, input_name, False)

    samples = model.sample(500)[0]
    samples[samples < samples_threshold] = 0
    samples = samples * 127
    cv2.imwrite(f"data/hmm_samples_{input_name}_{n_components}.png", samples.T * 2)

    print(samples)
    samples_data = np.rint(samples.T).astype(int).clip(0, 127)
    sample_encoded = restore(Encoded(samples_data, *encoded[1:]), start, end)
    m.to_midi(sample_encoded, f"data/predicted_{input_name}_{n_components}.midi")


if __name__ == '__main__':
    main()
