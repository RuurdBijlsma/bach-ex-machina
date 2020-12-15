import numpy as np
from matplotlib import pyplot as plt
from hmmlearn.hmm import GaussianHMM
import csv
from os import path


def fit_model(data, n_components):
    hmm = GaussianHMM(n_components=n_components)
    hmm.fit(data)
    return hmm, hmm.score(data)


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
            hmm, score = fit_model(data, n_components)
            avg_score += score
        # Score doesn't take complexity of model into account afaik,
        # but when I use Bayesian Information Criterion it says n_components=1 is best,
        # However visually this looks trash
        avg_score /= fit_iters
        scores.append((n_components, avg_score))
        if avg_score > best_score:
            print(f"Found new better model at n_components: {n_components}, score: {avg_score}")
            best_score = avg_score
            best_param = n_components, hmm

    print(best_score, best_param)
    _, hmm = best_param
    return hmm, scores


def main():
    input_data = np.loadtxt(path.join('input', 'F.txt'))
    hmm, _ = fit_model(input_data, 35)

    fig, axs = plt.subplots(2, 2)
    # hmm, scores = best_model(input_data)
    # axs[0, 1].set_title("Scores")
    # axs[0, 1].plot(scores)
    # print(scores)

    axs[0, 0].set_title("Data")
    axs[0, 0].plot(input_data)

    samples = hmm.sample(500)
    axs[1, 0].set_title("Samples")
    axs[1, 0].plot(samples[0])
    print(samples[0])

    plt.show()
    with open(path.join('input', 'out.txt'), mode='w', newline="") as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')

        for sample in samples[0]:
            notes = np.round(sample).astype(int)
            csv_writer.writerow(notes)


if __name__ == '__main__':
    main()
