import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage.filters import gaussian_filter1d
import sqlite3 as sql
import os
import pandas as pd
from datetime import datetime
from glob import glob

os.sys.path.append('../')
import utilities as u
import preprocessing as pp
import matplotlib.gridspec as gridspec



def single_session(sess):
    # load calcium data and aligned vr
    VRDat, C, A = pp.load_scan_sess(sess)

    # get trial by trial info
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    C_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(C,VRDat['pos']._values,VRDat['tstart']._values,VRDat['teleport']._values)
    C_morph_dict = u.trial_type_dict(C_trial_mat,trial_info['morphs'])
    occ_morph_dict = u.trial_type_dict(occ_trial_mat,trial_info['morphs'])

    # find place cells individually on odd and even trials
    # keep only cells with significant spatial information on both
    masks, FR, SI = u.place_cells_split_halves(C, VRDat['pos']._values,trial_info, VRDat['tstart']._values, VRDat['teleport']._values)

    # plot place cells by morph
    f_pc, ax_pc = u.plot_placecells(C_morph_dict,masks)

    ########################################################
    # number in each environment
    print('morph 0 place cells = %g out of %g , %f ' % (masks[0].sum(), masks[0].shape[0], masks[0].sum()/masks[0].shape[0]))
    print('morph 1 place cells = %g out of %g, %f' % (masks[1].sum(), masks[1].shape[0], masks[1].sum()/masks[1].shape[0]))


    # number with place fields in both
    common_pc = np.multiply(masks[0],masks[1])
    print('common place cells = %g' % common_pc.sum())
        # including, excluding reward zones

    # stability

    # width


    # reward cell scatter plot
    FR_0_cpc = FR[0]['all'][:,common_pc]
    FR_1_cpc = FR[1]['all'][:,common_pc]
    f_rc, ax_rc = reward_cell_scatterplot(FR_0_cpc,FR_1_cpc)

    # cell's topography
    # place cell in which morph

    # reward zone score

    # place field width

    # place cell reliability


    return FR, masks, SI


def reward_cell_scatterplot(fr0, fr1, rzone0 = [250,315], rzone1 = [350,415],tmax= 450):
    f = plt.figure(figsize=[10,10])
    gs = gridspec.GridSpec(5,5)
    ax = f.add_subplot(gs[0:-1,0:-1])


    #f,ax = plt.subplots()
    ax.scatter(5.*np.argmax(fr0,axis=0),5*np.argmax(fr1,axis=0),color='black')
    ax.plot(np.arange(tmax),np.arange(tmax),color='black')
    ax.fill_between(np.arange(tmax),rzone0[0],y2=rzone0[1],color=plt.cm.cool(0),alpha=.2)
    ax.fill_betweenx(np.arange(tmax),rzone0[0],x2=rzone0[1],color=plt.cm.cool(0),alpha=.2)
    ax.fill_between(np.arange(tmax),rzone1[0],y2=rzone1[1],color=plt.cm.cool(1.),alpha=.2)
    ax.fill_betweenx(np.arange(tmax),rzone1[0],x2=rzone1[1],color=plt.cm.cool(1.),alpha=.2)

    ax1 = f.add_subplot(gs[-1,0:-1])
    ax1.hist(5.*np.argmax(FR_0_cpc,axis=0),np.arange(0,tmax+10,10))
    ax1.fill_betweenx(np.arange(40),rzone0[0],x2=rzone0[1],color=plt.cm.cool(0),alpha=.2)
    ax1.fill_betweenx(np.arange(40),rzone1[0],x2=rzone1[1],color=plt.cm.cool(1.),alpha=.2)

    ax2 = f.add_subplot(gs[0:-1,-1])
    ax2.hist(5.*np.argmax(FR_1_cpc,axis=0),np.arange(0,tmax+10,10),orientation='horizontal')
    ax2.fill_between(np.arange(40),rzone0[0],y2=rzone0[1],color=plt.cm.cool(0),alpha=.2)
    ax2.fill_between(np.arange(40),rzone1[0],y2=rzone1[1],color=plt.cm.cool(1.),alpha=.2)

    return f, ax
