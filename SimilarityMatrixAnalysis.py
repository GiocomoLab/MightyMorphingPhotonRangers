import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import os
from datetime import datetime
from glob import glob

os.sys.path.append('../')
import utilities as u
import preprocessing as pp
import matplotlib.gridspec as gridspec



def single_session(sess, C= None, VRDat = None, zscore = True, spikes = False, normalize = False):
    '''calculate similarity matrices, average within the morphs and plot results'''
    # load calcium data and aligned vr
    if (C is None) or (VRDat is None) or (A is None):
        VRDat, C, Cd, S, A = pp.load_scan_sess(sess)

    if spikes:
        C = gaussian_filter1d(S,3,axis=0)

    if zscore:
        print(C.shape)
        Cz = sp.stats.zscore(C,axis=0)

    # get trial by trial info
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    print("sim script",tstart_inds.shape,teleport_inds.shape)
    C_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(C,VRDat['pos']._values,VRDat['tstart']._values,VRDat['teleport']._values)
    C_morph_dict = u.trial_type_dict(C_trial_mat,trial_info['morphs'])

    S = morph_simmat(C_morph_dict, normalize = normalize)
    m = len(C_morph_dict.keys())-3
    U, U_rnorm = morph_mean_simmat(S,m)

    f_S,ax_S = plot_simmat(S,m)

    f_U,ax_U = plt.subplots(2,1,figsize=[5,10])
    ax_U[0].imshow(U,cmap='Greys')

    ax_U[1].imshow(U_rnorm,cmap='Greys')

    return S, U, U_rnorm, (f_S,ax_S), (f_U, ax_U)


def plot_simmat(S,m):
    f,ax = plt.subplots(1,1, figsize=[m*2,m*2])
    # m = number of morphs
    N = S.shape[0]

    step = int(N/m)
    e = np.arange(step,N+1,step)

    ax.imshow(S,aspect='auto',cmap='Greys')
    ax.vlines(e,0,N,color='red',alpha=.2)
    ax.hlines(e,0,N,color='red',alpha=.2)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return f,ax

def morph_mean_simmat(S,m):
    ''''''
    # m = number of morphs
    N = S.shape[0]
    step = int(N/m)

    U = np.zeros([m,m])

    # average within morphs
    e = np.arange(0,N+1,step)
    edges = np.zeros([e.shape[0]-1,2])
    edges[:,0], edges[:,1] = e[:-1], e[1:]
    for i in range(m):
        for j in range(m):
            U[i,j] = S[int(edges[i,0]):int(edges[i,1]),int(edges[j,0]):int(edges[j,1])].ravel().mean()

    # normalize to make identity 1
    U_rnorm = np.zeros(U.shape)
    for z in range(U.shape[0]):
        U_rnorm[z,:] = U[z,:]/U[z,z]

    return U, U_rnorm

def morph_simmat(C_morph_dict, normalize = False):
    X = morph_by_cell_mat(C_morph_dict,normalize=normalize)
    return np.matmul(X.T,X)

def morph_by_cell_mat(C_morph_dict,normalize = False):
    k = 0
    for i,m in enumerate(C_morph_dict.keys()):

        if m not in ('all','labels','indices'):
            #print(m, C_morph_dict[m].keys())
            # firing rate maps
            fr = np.nanmean(C_morph_dict[m],axis=0)
            if normalize:
                for j in range(fr.shape[1]):
                    fr[:,j] = fr[:,j]/fr[:,j].sum()

            if k == 0:
                k+=1
                X = fr.T
            else:
                print(X.shape,fr.shape)
                X = np.hstack((X,fr.T))

    return X
