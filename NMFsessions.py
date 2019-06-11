import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn as sk
from sklearn.decomposition import NMF
import os
import pickle

import PlaceCellAnalysis as pc
import utilities as u
import preprocessing as pp
import behavior as b
import BayesianDecoding as bd
import ensemble as nmf

os.sys.path.append("C:\\Users\\mplitt\\MightyMorphingPhotonRangers\\CensoredLstsq")


def cross_val_ensemble_nmf(S_trial_mat,trial_info,dir="G:\\My Drive\\Figures\\TwoTower"):

    trialmask = np.zeros([S_flatmat.shape[0]])
    trialmask[np.random.permutation(S_flatmat.shape[0])[:int(S_flatmat.shape[0]/2)]]=1
    trialmask = trialmask>0.

    results = nmf.fit_ensemble(S_flatmat[trialmask,:],np.arange(1,11),n_replicates=5)

    return results
