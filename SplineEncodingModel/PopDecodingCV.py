import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import numpy as np
import matplotlib.pyplot as plt

import utilities as u
import preprocessing as pp
import behavior as b
import SimilarityMatrixAnalysis as sm
import scipy as sp
import PlaceCellAnalysis as pc
from SplineEncodingModel.LinearRegressionSpline import EncodingModel, NBDecodingModel
import matplotlib.gridspec as gridspec
import pickle


def run_sess(sess,ops_in={}):
    ops_out = {'fr':15.46,
                'outdir':os.path.join("G:\\My Drive\\Figures\\TwoTower\\PosCtxtSplineGLM",
                                    "%s_%s_%i" %(sess['MouseName'],sess['DateFolder'],sess['SessionNumber'])),
                'nfolds':3,
                'rthresh':.3}
    for k,v in ops_in.items():
        ops_out[k]=v


    VRDat,C, S, A = pp.load_scan_sess(sess)
    L_XC,Chat,cellmask = run_cross_val(C,VRDat,**ops_out)
    try:
        os.makedirs(ops_out['outdir'])
    except:
        pass

    with open(os.path.join(ops_out['outdir'],'mdloutput.pkl'),'wb') as f:
        pickle.dump({'Likelihood_XC':L_XC,'C_hat':Chat,'cellmask':cellmask},f)



    plot_likelihood(VRDat, L_XC,savefigs=True,outpath=ops_out['outdir'])



def run_cross_val(C,VRDat,fr=15.46,outdir=None,nfolds=3,rthresh=.3):
    '''run population decoding for '''
    C /=fr*100 # scale down to get reasonable "firing rate estimates"

    # get train and test masks for n-fold cross validation
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    randInds = np.random.permutation(tstart_inds.shape[0])
    test_mask = [np.zeros([C.shape[0],]) for i in range(nfolds)]
    for cnt,ind in enumerate(randInds.tolist()):
        test_mask[cnt%nfolds][tstart_inds[ind]:teleport_inds[ind]]+=1
    for i in range(nfolds):
        test_mask[i] = test_mask[i]>0

    effMorph =VRDat.morph._values+VRDat.bckgndJitter._values + VRDat.wallJitter._values
    effMorph = (effMorph+.25)/1.5

    # fit all cells to find those that are well fit by the model
    Chat = np.zeros(C.shape)
    Chat[:]=np.nan

    for i, test in enumerate(test_mask):
        print("training fold",i)
        test = test & (VRDat['pos']._values>0) & (VRDat['pos']._values<450)
        train = ~test & (VRDat['pos']._values>0) & (VRDat['pos']._values<450) & (VRDat['speed']._values>1)
        nb = NBDecodingModel()

        nb.poisson_fit(VRDat['pos']._values[train],effMorph[train],C[train,:])
        # predict on left out data
        Chat[test,:]=nb.poisson_predict_rate(VRDat['pos']._values[test],effMorph[test])

    # find cells for which the model fits well
    nanmask = np.isnan(Chat[:,0])
    Rvec = np.diagonal(1/(C.shape[0]-nanmask.sum())*np.matmul(sp.stats.zscore(C[~nanmask,:],axis=0).T,
                                                                sp.stats.zscore(Chat[~nanmask,:],axis=0)))
    cellmask = Rvec>rthresh
    print(cellmask.sum(), "cells included in decoding model")


    # get likelihood using same folds
    L_XC = np.zeros([C.shape[0],50,50]) # assume 50 x 50 sampling grid in model
    for i, test in enumerate(test_mask):
        print('start fold ', i)
        train = ~test & (VRDat['pos']._values>0) & (VRDat['pos']._values<450) & (VRDat['speed']._values>1)
        # init model
        nb = NBDecodingModel()
        C_fit = C[train,:]
        # train
        nb.poisson_fit(VRDat['pos']._values[train],effMorph[train],C_fit[:,cellmask])
        nb.poisson_predict_rate_decodingDM()

        # test on remaining time points
        C_decoding = C[test,:]
        C_decoding = C_decoding[:,cellmask]
        print('decoding fold', i)
        print("\t total timepoints =",test.sum())

        # fill in likelihood
        L_XC[test,:,:]= nb.poisson_decode(C_decoding)


    return L_XC, Chat, cellmask





def plot_likelihood(VRDat, L_XC,savefigs=True,outpath=None):


    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    effMorph =VRDat.morph._values+VRDat.bckgndJitter._values + VRDat.wallJitter._values
    effMorph = (effMorph+.25)/1.5

    if outpath is None:
        outpath = os.getcwd()

    # bin position
    pos_plot = VRDat['pos']/450.*50

    # get lick position
    lick_pos = u.lick_positions(VRDat.lick._values,VRDat.pos._values)/450.*50


    # get reward position
    reward_pos = u.lick_positions(VRDat.reward._values,VRDat.pos._values)/450.*50



    # plot single trial results
    singletrial_folder = os.path.join(outpath,"single_trials")
    try:
        os.makedirs(singletrial_folder)
    except:
        pass
    for j, (start,stop) in enumerate(zip(tstart_inds.tolist(),teleport_inds.tolist())):

        gs = gridspec.GridSpec(8,1)
        f_trial = plt.figure(figsize=[10,10])

        x = np.arange(stop-start)

        m_ax = f_trial.add_subplot(gs[0:4,:])
        m_ax.imshow(np.log(L_XC[start:stop,:,4:-4].sum(axis=1)).T,aspect='auto',cmap='cividis',zorder=0,alpha=.4)
        m_ax.scatter(x,effMorph[start:stop]*50,marker='o',color=plt.cm.cool(effMorph[start:stop].mean()),zorder=1)
        m_ax.set_xlim([0,x[-1]])

        pos_ax = f_trial.add_subplot(gs[4:,:])
        pos_ax.imshow(np.log(L_XC[start:stop,:,:].sum(axis=2)).T,aspect='auto',cmap='magma',zorder=0,alpha=.4)
        pos_ax.plot(x,pos_plot[start:stop],linewidth=2,zorder=1,alpha=1)
        pos_ax.scatter(x,lick_pos[start:stop],marker='x',color='green',zorder=2)
        pos_ax.scatter(x,reward_pos[start:stop],marker='o',color='blue',zorder=2)
        pos_ax.set_xlim([0,x[-1]])

        if savefigs:
            f_trial.savefig(os.path.join(singletrial_folder,'trial_%i.png' % j),format='png')



    # marginalize over position and plot trial by trial averaged log likelihood
    # for context (i.e. total log-likelihood for trial normalized by number of timepooints)
    trial_em = trial_info['morphs']+trial_info['towerJitter']+trial_info['wallJitter']+trial_info['bckgndJitter']
    trial_em+=.25
    trial_em/=1.5
    msort = np.argsort(trial_em)
    LLC_trial = np.zeros([tstart_inds.shape[0],50])
    for t,(start,stop) in enumerate(zip(tstart_inds.tolist(),teleport_inds.tolist())):
        LLC_trial[t,:] = np.log(L_XC[start:stop,:,:].sum(axis=1)).mean(axis=0)

    f_llc,ax_llc = plt.subplots()
    ax_llc.imshow(LLC_trial[msort,4:-4],aspect='auto',cmap='cividis')

    # same idea but plot probability and do it as a bunch of line plots with morph indicated by color
    f_pc,ax_pc = plt.subplots()
    for row in range(LLC_trial.shape[0]):
        ax_pc.plot(np.exp(LLC_trial[row,:]),color=plt.cm.cool(np.float(trial_em[row])), alpha = .3)


    if savefigs:
        f_llc.savefig(os.path.join(outpath,"context_loglikelihood.png"),format='png')
        f_pc.savefig(os.path.join(outpath,"context_proability.png"),format='png')
