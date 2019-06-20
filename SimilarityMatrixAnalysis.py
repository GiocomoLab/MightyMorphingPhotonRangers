import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import os
from datetime import datetime
from glob import glob
import sklearn as sk
import sklearn.cluster as clust


# os.sys.path.append('../')
import utilities as u
import preprocessing as pp
import matplotlib.gridspec as gridspec


def set_default_ops(d):
    ops={}
    ops['zscore']=False
    ops['deconv']=True
    ops['cell_normalize']=False
    ops['corr']=True
    ops['bootstrap']=False
    ops['mask'] = None
    for k,v in d.items():
        ops[k]=v

    return ops


def single_session(sess,ops = {},plot=True):
    '''calculate similarity matrices, average within the morphs and plot results'''
    # load calcium data and aligned vr
    VRDat, C, S, A = pp.load_scan_sess(sess,fneu_coeff=0.7)
    ops = set_default_ops(ops)

    if ops['deconv']:
        C=S
    else:
        C = u.df(C)


    if ops['mask'] is not None:
        C = C[:,ops['mask']]

    # get trial by trial info
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    C_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(C,VRDat['pos']._values,VRDat['tstart']._values,VRDat['teleport']._values)
    morphs = trial_info['morphs']
    C_morph_dict = u.trial_type_dict(C_trial_mat,morphs)

    mlist = np.unique(np.sort(morphs)).tolist()
    m = len(mlist)
    if ops['bootstrap']:
        nperms = 50
        S_full = morph_simmat(C_morph_dict, corr=ops['corr'])

        U_full = morph_mean_simmat(S_full,m)

        # allocate space
        S_bs = np.zeros([S_full.shape[0],S_full.shape[1], nperms])
        U_bs = np.zeros([m,m,nperms])

        for p in range(nperms):
            C_tmp = {}
            for morph in mlist:
                bs_n = int(.67*C_morph_dict[morph].shape[0]) # take random 2/3 of trials of each type
                order = np.random.permutation(C_morph_dict[morph].shape[0])[:bs_n]
                C_tmp[morph]= C_morph_dict[morph][order,:,:]


            S_bs[:,:,p] = morph_simmat(C_tmp,corr=corr,cell_normalize=cell_normalize)
            U_bs[:,:,p] = morph_mean_simmat(S_bs[:,:,p],m)

        f_S,ax_S = plot_simmat(np.nanmean(S_bs,axis=-1),m)

        f_U,ax_U = plt.subplots(figsize=[5,5])

        ax_U.imshow(np.nanmean(U_bs,axis=-1),cmap='Greys')

        return S_bs, U_bs, (f_S,ax_S), (f_U, ax_U)



    else:
        S = morph_simmat(C_morph_dict, corr=ops['corr'])


        U= morph_mean_simmat(S,m)
        # if m==5:
        #     for i,morph in enumerate([0,.25,.5,.75,1.]):
        #         if i in (0,1):
        #             FR = C_morph_dict[morph]
        #             FR0 = np.nanmean(FR[0::2,:,:],axis=0)
        #             FR1 = np.nanmean(FR[1::2,:,:],axis=0)
        #             FR0,FR1 = sp.stats.zscore(FR0.ravel()),sp.stats.zscore(FR1.ravel())
        #             U[i,i] = 1/FR0.shape[0]*np.dot(FR0,FR1)
        #
        #         # U[i,i] = (1/FR0.shape[0]*np.matmul(FR0.T,FR1)).mean()

        if plot:
            f_S,ax_S = plot_simmat(S,m)

            f_U,ax_U = plt.subplots(figsize=[5,5])
            ax_U.imshow(U,cmap='Greys')

            # ax_U[1].imshow(U_rnorm,cmap='Greys')

            return S, U, (f_S,ax_S), (f_U, ax_U)
        else:
            return S, U


def plot_trial_simmat(C,trial_info,vmax=None):
    '''plot similarity matrices comparing each trial'''

    if vmax is None:
        vmax = np.percentile(C.ravel(),95)


    f = plt.figure(figsize=[30,12])
    gs = gridspec.GridSpec(14,30)

    effMorph = trial_info['morphs'] +  trial_info['wallJitter'] + trial_info['bckgndJitter']
    msort = np.argsort(effMorph)

    x=np.arange(effMorph.size)
    rmask = trial_info['rewards']>0
    tnumber = np.arange(x.shape[0])/x.shape[0]


    # sort by trial order
    c_ax = f.add_subplot(gs[:10,:10])
    c_ax.imshow(C,cmap='magma',vmin=0,vmax=vmax,aspect='auto')
    c_ax.set_yticks([])
    c_ax.set_xticks([])

    m_ax = f.add_subplot(gs[10:12,:10])
    m_ax.scatter(x,effMorph,c=effMorph,cmap='cool')
    m_ax.scatter(x[~rmask],effMorph[~rmask],c='black',s=10)
    m_ax.set_xlim([0,x.shape[0]])
    m_ax.set_yticks([])
    m_ax.set_xticks([])

    t_ax = f.add_subplot(gs[12:,:10])
    t_ax.scatter(x,tnumber,c=tnumber,cmap='viridis')
    t_ax.scatter(x[~rmask],tnumber[~rmask],c='black',s=10)
    t_ax.set_xlim([0,x.shape[0]])
    t_ax.set_yticks([])

    # sort similarity matrix by morph
    C_msort = _sort_simmat(C,msort)

    # sort by morph value
    cm_ax = f.add_subplot(gs[:10,10:20])
    cm_ax.imshow(C_msort,cmap='magma',vmin=0,vmax=vmax,aspect='auto')
    cm_ax.set_yticks([])
    cm_ax.set_xticks([])

    mm_ax = f.add_subplot(gs[10:12,10:20])
    mm_ax.scatter(x,effMorph[msort],c=effMorph[msort],cmap='cool')
    emr = np.copy(effMorph)
    emr[rmask]=np.nan

    mm_ax.scatter(x,emr[msort],c='black',s=10)
    mm_ax.set_xlim([0,x.shape[0]])
    mm_ax.set_yticks([])
    mm_ax.set_xticks([])
    tm_ax = f.add_subplot(gs[12:,10:20])
    tm_ax.scatter(x,tnumber[msort],c=tnumber[msort],cmap='viridis')
    tm_ax.set_xlim([0,x.shape[0]])
    tm_ax.set_yticks([])

    # sort similarity matrix by cluster - laplacian eigenmaps
    clustsort = _sort_clusters(cluster_simmat(C),effMorph)
    C_csort = _sort_simmat(C,clustsort)


    cc_ax = f.add_subplot(gs[:10,20:])
    cc_ax.imshow(C_csort,cmap='magma',vmin=0,vmax=vmax,aspect='auto')
    cc_ax.set_yticks([])
    cc_ax.set_xticks([])
    mc_ax = f.add_subplot(gs[10:12,20:])
    mc_ax.scatter(x,effMorph[clustsort],c=effMorph[clustsort],cmap='cool')
    mc_ax.scatter(x,emr[clustsort],c='black',s=10)
    mc_ax.set_yticks([])
    mc_ax.set_xticks([])
    mc_ax.set_xlim([0,x.shape[0]])
    tc_ax = f.add_subplot(gs[12:,20:])
    tc_ax.scatter(x,tnumber[clustsort],c=tnumber[clustsort],cmap='viridis')
    tc_ax.set_xlim([0,x.shape[0]])
    tc_ax.set_yticks([])

    return f, [[c_ax,m_ax,t_ax],[cm_ax,mm_ax,tm_ax],[cc_ax,mc_ax,tc_ax]]

def cluster_simmat(C):
    score = []
    for c in range(2,10):
        spectclust = clust.SpectralClustering(n_clusters=c,affinity='precomputed')
        labels = spectclust.fit_predict(C)
        s=sk.metrics.silhouette_score(1-C,labels,metric='precomputed')
        score.append(np.floor(100.*s))
        print(s*100.)

    c = np.argmax(score)+2
    spectclust = clust.SpectralClustering(n_clusters=c,affinity='precomputed')
    spectclust.fit(C)
    return spectclust.labels_

def _sort_clusters(clustlabels,metric):

    nc = np.unique(clustlabels).shape[0]
    clustmean = np.array([metric[clustlabels==i].mean() for i in range(nc)])
    clusterOrder = np.argsort(clustmean)
    labels = np.zeros(metric.shape)

    for i,cl in enumerate(clusterOrder.tolist()):
        labels[clustlabels==cl]=i

    return np.argsort(labels)


def _sort_simmat(A,sort):
    A = A[sort,:]
    return A[:,sort]


def trial_simmat(S_tm):

    # smooth single cell firing rate
    S_mat = S_tm.reshape([S_tm.shape[0],-1])
    # normalize by L2 norm
    S_mat/=np.linalg.norm(S_mat,ord=2,axis=1)[:,np.newaxis]

    return np.matmul(S_mat,S_mat.T)



def plot_simmat(S,m):
    f,ax = plt.subplots(1,1, figsize=[m*2,m*2])
    # m = number of morphs
    N = S.shape[0]

    step = int(N/m)
    e = np.arange(step,N+1,step)-1

    ax.imshow(S,aspect='auto',cmap='Greys')
    ax.vlines(e,0,N,color='red',alpha=.2)
    ax.hlines(e,0,N,color='red',alpha=.2)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return f,ax

def morph_mean_simmat(SM,m):
    ''''''
    # m = number of morphs
    N = SM.shape[0]
    step = int(N/m)

    U = np.zeros([m,m])

    # average within morphs
    e = np.arange(0,N+1,step)
    # take off diagonal in each block
    for i in range(m):
        row_slice = np.arange(e[i],e[i+1])
        for j in range(m):
            col_slice= np.arange(e[j],e[j+1])
            U[i,j]= SM[row_slice,col_slice].ravel().mean()
    return U


def morph_simmat(C_morph_dict, corr = False ):
    C_dict0,C_dict1 = {},{}


    X = morph_by_cell_mat(C_morph_dict)

    if corr: # zscore to get correlation
        X=sp.stats.zscore(X,axis=0)
        return 1/X.shape[0]*np.matmul(X.T,X)

    else: # scale by l2 norm to give cosine similarity
        X/=np.power(X,2).sum(axis=0)[np.newaxis,:]
        return np.matmul(X.T,X)


def morph_by_cell_mat(C_morph_dict,sig=3):
    k = 0
    for i,m in enumerate(C_morph_dict.keys()):
        if m not in ('all','labels','indices'):
            #print(m, C_morph_dict[m].keys())
            # firing rate maps
            fr = np.nanmean(C_morph_dict[m],axis=0)
            fr = gaussian_filter(fr,[sig,0])
            if k == 0:
                k+=1
                X = fr.T
            else:
                print(X.shape,fr.shape)
                X = np.hstack((X,fr.T))
    return X
