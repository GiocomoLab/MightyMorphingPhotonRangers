import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import sqlite3 as sql
import os
import pandas as pd
from datetime import datetime
from glob import glob

os.sys.path.append('../')
import utilities as u
import preprocessing as pp
import matplotlib.gridspec as gridspec



def single_session(sess, C= None, VRDat = None, A=None,savefigs = False,fbase=None):
    # load calcium data and aligned vr
    if (C is None) and (VRDat is None) and (A is None):
        VRDat, C, Cd, S, A = pp.load_scan_sess(sess)

    # get trial by trial info
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    C_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(C,VRDat['pos']._values,VRDat['tstart']._values,VRDat['teleport']._values)
    C_morph_dict = u.trial_type_dict(C_trial_mat,trial_info['morphs'])
    occ_morph_dict = u.trial_type_dict(occ_trial_mat,trial_info['morphs'])

    # find place cells individually on odd and even trials
    # keep only cells with significant spatial information on both
    masks, FR, SI = place_cells_split_halves(C, VRDat['pos']._values,trial_info, VRDat['tstart']._values, VRDat['teleport']._values)

    # plot place cells by morph
    f_pc, ax_pc = plot_placecells(C_morph_dict,masks)

    ########################################################
    # number in each environment
    print('morph 0 place cells = %g out of %g , %f ' % (masks[0].sum(), masks[0].shape[0], masks[0].sum()/masks[0].shape[0]))
    print('morph 1 place cells = %g out of %g, %f' % (masks[1].sum(), masks[1].shape[0], masks[1].sum()/masks[1].shape[0]))


    # number with place fields in both
    common_pc = np.multiply(masks[0],masks[1])
    print('common place cells = %g' % common_pc.sum())
        # including, excluding reward zones

    ####### stability
    # first vs second half correlation
    sc_corr, pv_corr= {}, {}
    sc_corr[0], pv_corr[0] = stability_split_halves(C_morph_dict[0])
    sc_corr[1], pv_corr[1] = stability_split_halves(C_morph_dict[1])

    #   (fancier version, tortuosity of warping function over time)
    # not implemented yet

    ####### tuning specificity
    #   vector length of circularized tuning curve
    mvl = {}
    mvl[0] = meanvectorlength(FR[0]['all'])
    mvl[1] = meanvectorlength(FR[1]['all'])

    # reward cell scatter plot
    FR_0_cpc = FR[0]['all'][:,common_pc]
    FR_1_cpc = FR[1]['all'][:,common_pc]
    f_rc, ax_rc = reward_cell_scatterplot(FR_0_cpc,FR_1_cpc)

    # cell's topography
    # place cell in which morph
    both = np.where((masks[0]>0) & (masks[1]>0) )[0]
    none = np.where((masks[0]==0) & (masks[1]==0))[0]
    m0 = np.where((masks[0]==1) & (masks[1]==0))[0]
    m1 = np.where((masks[0]==0) & (masks[1]==1))[0]
    tvals = np.zeros([A.shape[1],])
    tvals[both]=.01
    tvals[m0]=-1
    tvals[m1]=1

    # reward zone score

    # place field width

    # place cell reliability

    if savefigs:
        f_pc.savefig(fbase+"_pc.pdf",format = 'pdf')
        f_pc.savefig(fbase+"_pc.svg",format = 'svg')

        f_rc.savefig(fbase+"_rc.pdf",format = 'pdf')
        f_rc.savefig(fbase+"_rc.svg",format = 'svg')


    return FR, masks, SI

def cell_topo_plot(A_k,vals,fov=[512,796],map = 'cool', min = -1, max = 1):
    ''' given sparse matrix of cell footprints, A_k, and values associated with
    cells, vals, plot shape of cells colored by vals'''

    nz = A_k.nonzero()
    A= np.zeros(A_k.shape)
    A[nz]=1

    for i, v in enumerate(vals.tolist()):
        A[:,i]*=v

    A_m = np.ma.array(A.max(axis=1) + A.min(axis=1))
    A_m[A_m==0]=np.nan

    f, ax = plt.subplots(figsize=[15,15])
    ax.imshow(np.reshape(A_m,fov,order='F'),cmap=map,vmin=min,vmax=max)

    return A_m, (f,ax)

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
    ax1.hist(5.*np.argmax(fr0,axis=0),np.arange(0,tmax+10,10))
    ax1.fill_betweenx(np.arange(40),rzone0[0],x2=rzone0[1],color=plt.cm.cool(0),alpha=.2)
    ax1.fill_betweenx(np.arange(40),rzone1[0],x2=rzone1[1],color=plt.cm.cool(1.),alpha=.2)

    ax2 = f.add_subplot(gs[0:-1,-1])
    ax2.hist(5.*np.argmax(fr1,axis=0),np.arange(0,tmax+10,10),orientation='horizontal')
    ax2.fill_between(np.arange(40),rzone0[0],y2=rzone0[1],color=plt.cm.cool(0),alpha=.2)
    ax2.fill_between(np.arange(40),rzone1[0],y2=rzone1[1],color=plt.cm.cool(1.),alpha=.2)

    return f, ax

def stability_split_halves(trial_mat):
    '''calculate first half vs second half tuning curve correlation'''

    # assume trial_mat is (trials x pos x cells)
    half = int(trial_mat.shape[0]/2)

    fr0 = np.squeeze(np.nanmean(trial_mat[:half,:,:],axis=0))
    fr1 = np.squeeze(np.nanmean(trial_mat[half:,:,:],axis=0))

    sc_corr, pv_corr = stability(fr0,fr1)
    return sc_corr, pv_corr

def stability(fr0, fr1):
    # single cell place cell correlations
    sc_corr = np.array([sp.stats.pearsonr(fr0[:,cell],fr1[:,cell]) for cell in range(fr0.shape[1])])

    # population vector correlation
    pv_corr  = np.array([sp.stats.pearsonr(fr0[pos,:],fr1[pos,:]) for pos in range(fr0.shape[0])])
    return sc_corr, pv_corr

def meanvectorlength(fr):
    return np.linalg.norm(fr-fr.mean())

def spatial_info(frmap,occupancy):
    '''calculate spatial information bits/spike'''
    ncells = frmap.shape[1]

    SI = []
    #p_map = np.zeros(frmap.shape)
    for i in range(ncells):
        p_map = gaussian_filter(frmap[:,i],2)
        p_map /= p_map.sum()
        denom = np.multiply(p_map,occupancy).sum()


        si = 0
        for c in range(frmap.shape[0]):
            if (p_map[c]<0) or (occupancy[c]<0):
                print("we have a problem")
            if (p_map[c] >= 0) and (occupancy[c]>=0):
                #print(p_map[c],denom,np.log2(p_map[c]/denom))

                si+= occupancy[c]*p_map[c]*np.log2(p_map[c]/denom)
            #print(p_)
        SI.append(si)

    return np.array(SI)


def place_cells_split_halves(C, position, trial_info, tstart_inds, teleport_inds):
    '''get masks for significant place cells that have significant place info
    in both even and odd trials'''

    C_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(C,position,tstart_inds,teleport_inds)
    C_morph_dict = u.trial_type_dict(C_trial_mat,trial_info['morphs'])
    occ_morph_dict = u.trial_type_dict(occ_trial_mat,trial_info['morphs'])
    tstart_inds, teleport_inds = np.where(tstart_inds==1)[0], np.where(teleport_inds==1)[0]
    tstart_morph_dict = u.trial_type_dict(tstart_inds,trial_info['morphs'])
    teleport_morph_dict = u.trial_type_dict(teleport_inds,trial_info['morphs'])

    # for each morph value
    FR,masks,SI = {}, {}, {}
    for m in [0, 1]:

        FR[m]= {}
        SI[m] = {}

        # firing rate maps
        FR[m]['all'] = np.nanmean(C_morph_dict[m],axis=0)
        FR[m]['odd'] = np.nanmean(C_morph_dict[m][0::2,:,:],axis=0)
        FR[m]['even'] = np.nanmean(C_morph_dict[m][1::2,:,:],axis=0)

        # occupancy
        occ_o, occ_e = occ_morph_dict[m][0::2,:].sum(axis=0), occ_morph_dict[m][1::2,:].sum(axis=0)
        occ_o/=occ_o.sum()
        occ_e/=occ_e.sum()
        occ_all = occ_morph_dict[m].sum(axis=0)
        occ_all /= occ_all.sum()

        SI[m]['all'] =  spatial_info(FR[m]['all'],occ_all)
        SI[m]['odd'] = spatial_info(FR[m]['odd'],occ_o)
        SI[m]['even'] = spatial_info(FR[m]['even'],occ_e)



        p_e, shuffled_SI = spatial_info_perm_test(SI[m]['even'],C,position,tstart_morph_dict[m][1::2],teleport_morph_dict[m][1::2],nperms=100)
        p_o, shuffled_SI = spatial_info_perm_test(SI[m]['odd'],C,position,tstart_morph_dict[m][0::2],teleport_morph_dict[m][0::2], nperms = 100 ) #,shuffled_SI=shuffled_SI)
        #for i in range(SI[m]['all'].shape[0]):
        #    print("%d: SI odd %.2E SI even %.2E, p_o %.2E p_e %.2E m %r" %(i,SI[m]['odd'][i],SI[m]['even'][i],p_e[i],p_o[i],(p_e[i]>.95) * (p_o[i]>.95)))

        masks[m]=np.multiply(p_e>.95,p_o>.95)

    return masks, FR, SI



def spatial_info_perm_test(SI,C,position,tstart,tstop,nperms = 10000,shuffled_SI=None):
    '''run permutation test on spatial information calculations. returns empirical p-values for each cell'''
    if len(C.shape)>2:
        C = np.expand_dims(C,1)

    if shuffled_SI is None:
        shuffled_SI = np.zeros([nperms,C.shape[1]])

        for perm in range(nperms):
            #C_perm = np.roll(C,randrange(position.shape[0]),axis=0)
            C_tmat, occ_tmat, edes,centers = u.make_pos_bin_trial_matrices(C,position,tstart,tstop,perm=True)
            fr, occ = np.squeeze(np.nanmean(C_tmat,axis=0)), occ_tmat.sum(axis=0)
            occ/=occ.sum()
            #pos_perm = np.roll(position,randrange(position.shape[0]))
            #fr,occ = rate_map(C,pos_perm,bin_size=5)
            #fr, occ = rate_map(C_perm,position,bin_size=5)
            si = spatial_info(fr,occ)
            shuffled_SI[perm,:] = si


    p = np.zeros([C.shape[1],])
    for cell in range(C.shape[1]):
        #print(SI[cell],np.max(shuffled_SI[:,cell]))
        #p[cell] = np.where(SI[cell]>shuffled_SI[:,cell])[0].shape[0]/nperms
        p[cell] = np.sum(SI[cell]>shuffled_SI[:,cell])/nperms

    return p, shuffled_SI

def plot_placecells(C_morph_dict,masks):
    '''plot place place cell results'''

    morphs = [k for k in C_morph_dict.keys() if isinstance(k,np.float64)]
    f,ax = plt.subplots(2,len(morphs),figsize=[5*len(morphs),15])

    getSort = lambda fr : np.argsort(np.argmax(np.squeeze(np.nanmean(fr,axis=0)),axis=0))
    sort0 = getSort(C_morph_dict[0][:,:,masks[0]])
    #print(masks[0].shape,sort0.shape)
    #print(sort0)
    sort1 = getSort(C_morph_dict[1][:,:,masks[1]])

    for i,m in enumerate(morphs):
        fr = np.squeeze(np.nanmean(C_morph_dict[m],axis=0))
        fr_n = np.copy(fr)
        for j in range(fr.shape[1]):
            fr_n[:,j] = gaussian_filter1d(fr[:,j]/fr[:,j].max(),2)
            #fr_n[:,j] = gaussian_filter1d(fr[:,j],2)
        fr_n0, fr_n1 = fr_n[:,masks[0]], fr_n[:,masks[1]]
        fr_n0, fr_n1 = fr_n0[:,sort0], fr_n1[:,sort1]
        ax[0,i].imshow(fr_n0.T,aspect='auto',cmap='Greys')
        ax[1,i].imshow(fr_n1.T,aspect='auto',cmap='Greys')

    return f, ax
