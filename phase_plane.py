import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SVMSMOTE

import common.constants as gv
from common.options import set_options
from common.plot_utils import save_fig, pkl_save

from data.get_data import get_X_y_days, get_X_y_S1_S2
from preprocess.helpers import avg_epochs
from preprocess.augmentation import spawner

from decode.classifiers import get_clf
from decode.coefficients import get_coefs

from statistics.bootstrap import my_boots_ci


def get_overlap(X, y, coefs):

    if coefs.ndim > 1:
        overlap = np.zeros((X.shape[-1], X.shape[0], 4))
    else:
        overlap = np.zeros((X.shape[-1], X.shape[0]))

    print("overlap", overlap.shape)

    for i_epoch in range(X.shape[-1]):
        overlap[i_epoch] = np.dot(coefs, X[..., i_epoch].T).T

    idx = np.where(y == 0)[0]
    overlap_A = -overlap[:, idx] / X.shape[1]
    overlap_B = -overlap[:, ~idx] / X.shape[1]

    overlap = np.stack((overlap_A, overlap_B))
    return overlap
    # return -overlap / X.shape[1]  # normalized by number of neurons


def get_overlap_trials(day, overlap, features="sample"):

    options = set_options()

    options["day"] = day
    options["overlap"] = overlap

    X_days, y_days = get_X_y_days(IF_PREP=1, IF_RELOAD=False)

    model = get_clf(**options)

    options["task"] = "Dual"

    if options["overlap"].lower() == "sample":
        options["features"] = "sample"
        options["task"] = ""
    else:
        options["features"] = "distractor"

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print(X_S1_S2.shape, y_S1_S2.shape)

    if options["overlap"].lower() == "sample":
        X_avg = avg_epochs(X_S1_S2, epochs=["ED"])
    else:  # distractor
        X_avg = avg_epochs(X_S1_S2, epochs=["MD"])

    coefs = get_coefs(model, X_avg, y_S1_S2, **options)

    print(
        "trials", X_S1_S2.shape[0], "coefs", coefs.shape, "non_zero", np.sum(coefs != 0)
    )

    options["task"] = "DualGo"
    options["features"] = "sample"
    options["trials"] = "correct"

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print(X_S1_S2.shape, y_S1_S2.shape)

    overlap = get_overlap(X_S1_S2, y_S1_S2, coefs)

    return overlap


def plot_kappa_plane(day="first"):

    overlap_sample = get_overlap_trials(day, overlap="sample")
    overlap_dist = get_overlap_trials(day, overlap="distractor")

    # sample_avg = avg_epochs(overlap_sample.T, epochs=["LD"])
    # dist_avg = avg_epochs(overlap_dist.T, epochs=["LD"])

    # print(sample_avg.shape)
    # print(dist_avg.shape)

    return overlap_sample, overlap_dist

    # radius, theta = carteToPolar(sample_avg, dist_avg)

    # # plt.figure("overlaps_plane_" + day)
    # # plt.plot(np.cos(theta), np.sin(theta), "o")
    # # plt.xlabel("Sample Overlap")
    # # plt.ylabel("Dist. Overlap")

    # plot_phase_dist(day, theta)


def carteToPolar(x, y):
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    return radius, theta * 180 / np.pi


def plot_phase_dist(day, theta):

    plt.figure("overlaps_phases_" + day)
    plt.hist(theta % 180, histtype="step", density=1, bins="auto")
    plt.xlim([0, 180])
    plt.xticks([0, 45, 90, 135, 180])

    plt.xlabel("Overlaps Pref. Dir. (°)")
    plt.ylabel("Density")


if __name__ == "__main__":

    overlap_sample, overlap_dist = plot_kappa_plane("first")
    # overlap_sample, overlap_dist = plot_kappa_plane("last")
    # plot_kappa_plane("last")
