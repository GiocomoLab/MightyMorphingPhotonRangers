import numpy as np
import h5py
import scipy as sp
import scipy.stats
import scipy.io
import scipy.interpolate
from random import randrange
import sqlite3 as sql
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
from datetime import datetime
from glob import glob
import os.path
from astropy.convolution import convolve, Gaussian1DKernel


def spatial_info(frmap,occupancy):
    '''calculate spatial information bits/spike'''
    ncells = frmap.shape[1]

    SI = []
    #p_map = np.zeros(frmap.shape)
    for i in range(ncells):
        p_map = gaussian_filter(frmap[:,i],2)
        p_map /= p_map.sum()
        #p_map = gaussian_filter(frmap[:,i],2)/frmap[:,i].sum()
        #p_map = np.squeeze(frmap[:,i]/frmap[:,i].sum())
        denom = np.multiply(p_map,occupancy).sum()
        #print(denom)

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


def rate_map(C,position,bin_size=10,min_pos = 0, max_pos=450):
    '''non-normalized rate map E[df/F]|_x '''
    bin_edges = np.arange(min_pos,max_pos+bin_size,bin_size).tolist()
    if len(C.shape) ==1:
        C = np.expand_dims(C,axis=1)
    frmap = np.zeros([len(bin_edges)-1,C.shape[1]])
    frmap[:] = np.nan
    occupancy = np.zeros([len(bin_edges)-1,])
    for i, (edge1,edge2) in enumerate(zip(bin_edges[:-1],bin_edges[1:])):
        if np.where((position>edge1) & (position<=edge2))[0].shape[0]>0:
            frmap[i] = C[(position>edge1) & (position<=edge2),:].mean(axis=0)
            occupancy[i] = np.where((position>edge1) & (position<=edge2))[0].shape[0]
        else:
            pass
    return frmap, occupancy/occupancy.ravel().sum()


def place_cells_split_halves(C, position, trial_info, tstart_inds, teleport_inds):
    '''get masks for significant place cells that have significant place info
    in both even and odd trials'''

    C_trial_mat, occ_trial_mat, edges,centers = make_pos_bin_trial_matrices(C,position,tstart_inds,teleport_inds)
    C_morph_dict = trial_type_dict(C_trial_mat,trial_info['morphs'])
    occ_morph_dict = trial_type_dict(occ_trial_mat,trial_info['morphs'])
    tstart_inds, teleport_inds = np.where(tstart_inds==1)[0], np.where(teleport_inds==1)[0]
    tstart_morph_dict = trial_type_dict(tstart_inds,trial_info['morphs'])
    teleport_morph_dict = trial_type_dict(teleport_inds,trial_info['morphs'])

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


        p_e, shuffled_SI = spatial_info_perm_test(SI[m]['even'],C,position,tstart_morph_dict[m],teleport_morph_dict[m],nperms=100)
        p_o, shuffled_SI = spatial_info_perm_test(SI[m]['odd'],C,position,tstart_morph_dict[m],teleport_morph_dict[m],shuffled_SI=shuffled_SI)

        masks[m]=np.multiply(p_e>.95,p_o<.95)

    return masks, FR, SI



def spatial_info_perm_test(SI,C,position,tstart,tstop,nperms = 10000,shuffled_SI=None):
    '''run permutation test on spatial information calculations. returns empirical p-values for each cell'''
    if len(C.shape)>2:
        C = np.expand_dims(C,1)

    if shuffled_SI is None:
        shuffled_SI = np.zeros([nperms,C.shape[1]])

        for perm in range(nperms):
            #C_perm = np.roll(C,randrange(position.shape[0]),axis=0)
            C_tmat, occ_tmat, edes,centers = make_pos_bin_trial_matrices(C,position,tstart,tstop,perm=True)
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


def make_pos_bin_trial_matrices(arr, pos, tstart, tstop,method = 'mean',bin_size=5,perm=False):
    '''make a ntrials x position x neurons tensor'''
    if tstart.max()<=1: # if binary, leaving in for backwards compatibility
        tstart_inds, tstop_inds = np.where(tstart==1)[0],np.where(tstop==1)[0]
        ntrials = np.sum(tstart)
    else:
        tstart_inds, tstop_inds = tstart, tstop
        ntrials = tstart.shape[0]

    #ntrials = np.sum(tstart)
    bin_edges = np.arange(0,450+bin_size,bin_size)
    bin_centers = bin_edges[:-1]+bin_size/2
    bin_edges = bin_edges.tolist()
    #print(len(bin_edges),bin_centers.shape)


    if len(arr.shape)<2:
        arr = np.expand_dims(arr,axis=1)

    trial_mat = np.zeros([int(ntrials),len(bin_edges)-1,arr.shape[1]])
    trial_mat[:] = np.nan
    occ_mat = np.zeros([int(ntrials),len(bin_edges)-1])



    for trial in range(int(ntrials)):

            firstI, lastI = tstart_inds[trial], tstop_inds[trial]
            arr_t,pos_t = arr[firstI:lastI,:], pos[firstI:lastI]
            if perm:
                pos_t = np.roll(pos_t,np.random.randint(pos_t.shape[0]))
            map, occ = rate_map(arr_t,pos_t,bin_size=bin_size)
            trial_mat[trial,:,:] = map
            occ_mat[trial,:] = occ
            #print(map.ravel())
    # self.trial_matrices = trial_matrices
    return np.squeeze(trial_mat), np.squeeze(occ_mat), bin_edges, bin_centers

def cnmf_com(A,d1,d2,d3):
    '''returns center of mass of cells given spatial footprints and native dimensions'''
    pass
    # return centers

def trial_tensor(C,labels,trig_inds,pre=50,post=50):
    '''create a tensor of trial x time x neural dimension for arbitrary centering indices'''

    if len(C.shape)==1:
        trialMat = np.zeros([trig_inds.shape[0],pre+post,1])
        C = np.expand_dims(C,1)
    else:
        trialMat = np.zeros([trig_inds.shape[0],pre+post,C.shape[1]])
    labelVec = np.zeros([trig_inds.shape[0],])

    for ind, t in enumerate(trig_inds):
        labelVec[ind] = labels[t]

        if t-pre <0:
            trialMat[ind,pre-t:,:] = C[0:t+post,:]
            trialMat[ind,0:pre-t,:] = C[0,:]

        elif t+post>C.shape[0]:
            print(trialMat.shape)
            print(t, post)
            print(C.shape[0])
            print(C[t-pre:,0].shape)

            trialMat[ind,:C.shape[0]-t-post,:] = C[t-pre:,:]
            trialMat[ind,C.shape[0]-t-post:,:] = C[-1,:]

        else:
            trialMat[ind,:,:] = C[t-pre:t+post,:]

    return trialMat, labelVec

def across_trial_avg(trialMat,labelVec):
    '''use output of trial_tensor function to return trial average'''
    labels = np.unique(labelVec)

    if len(trialMat.shape)==3:
        avgMat = np.zeros([labels.shape[0],trialMat.shape[1],trialMat.shape[2]])
    else:
        avgMat = np.zeros([labels.shape[0],trialMat.shape[1],1])

    for i, val in enumerate(labels.tolist()):
        #print(np.where(labelVec==val)[0].shape)
        avgMat[i,:,:] = np.nanmean(trialMat[labelVec==val,:,:],axis=0)

    return avgMat, labels



def trial_type_dict(mat,type_vec):
    '''make dictionary where each key is a trial type and data is arbitrary trial x var x var data
    should be robust to whether or not non-trial dimensions exist'''
    d = {'all': np.squeeze(mat)}
    ndim = len(d['all'].shape)
    d['labels'] = type_vec
    d['indices']={}
    for i,m in enumerate(np.unique(type_vec)):
        d['indices'][m] = np.where(type_vec==m)[0]

        if ndim==1:
            d[m] = d['all'][d['indices'][m]]
        elif ndim==2:
            d[m] = d['all'][d['indices'][m],:]
        elif ndim==3:
            d[m] = d['all'][d['indices'][m],:,:]
        else:
            raise(Exception("trial matrix is incorrect dimensions"))

    return d



def by_trial_info(data,rzone0=(250,315),rzone1=(350,415)):
    '''get abunch of single trial behavioral information and save it in a dictionary'''
    tstart_inds, teleport_inds = data.index[data.tstart==1],data.index[data.teleport==1]
    #print(tstart_inds.shape[0],teleport_inds.shape[0])
    trial_info={}
    morphs = np.zeros([tstart_inds.shape[0],])
    max_pos = np.zeros([tstart_inds.shape[0],])
    rewards = np.zeros([tstart_inds.shape[0],])
    zone0_licks = np.zeros([tstart_inds.shape[0],])
    zone1_licks = np.zeros([tstart_inds.shape[0],])
    zone0_speed = np.zeros([tstart_inds.shape[0],])
    zone1_speed = np.zeros([tstart_inds.shape[0],])
    pcnt = np.zeros([tstart_inds.shape[0],]); pcnt[:] = np.nan
    wallJitter= np.zeros([tstart_inds.shape[0],])
    towerJitter= np.zeros([tstart_inds.shape[0],])
    bckgndJitter= np.zeros([tstart_inds.shape[0],])
    clickOn= np.zeros([tstart_inds.shape[0],])
    pos_lick = np.zeros([tstart_inds.shape[0],])
    pos_lick[:] = np.nan
    for (i,(s,f)) in enumerate(zip(tstart_inds,teleport_inds)):
        sub_frame = data[s:f]
        m, counts = sp.stats.mode(sub_frame['morph'],nan_policy='omit')
        if len(m)>0:
            morphs[i] = m
            max_pos[i] = np.nanmax(sub_frame['pos'])
            rewards[i] = np.nansum(sub_frame['reward'])
            zone0_mask = (sub_frame.pos>=rzone0[0]) & (sub_frame.pos<=rzone0[1])
            zone1_mask = (sub_frame.pos>=rzone1[0]) & (sub_frame.pos<=rzone1[1])
            zone0_licks[i] = np.nansum(sub_frame.loc[zone0_mask,'lick'])
            zone1_licks[i] = np.nansum(sub_frame.loc[zone1_mask,'lick'])
            zone0_speed[i]=np.nanmean(sub_frame.loc[zone0_mask,'speed'])
            zone1_speed[i] = np.nanmean(sub_frame.loc[zone1_mask,'speed'])
            wj, c = sp.stats.mode(sub_frame['wallJitter'],nan_policy='omit')
            wallJitter[i] = wj
            tj, c = sp.stats.mode(sub_frame['towerJitter'],nan_policy='omit')
            towerJitter[i] = tj
            bj, c = sp.stats.mode(sub_frame['bckgndJitter'],nan_policy='omit')
            bckgndJitter = bj
            co, c = sp.stats.mode(sub_frame['clickOn'],nan_policy='omit')
            clickOn[i]=co

            lick_mask = sub_frame.lick>0
            pos_lick_mask = lick_mask & (zone0_mask | zone1_mask)
            pos_licks = sub_frame.loc[pos_lick_mask,'pos']
            if pos_licks.shape[0]>0:
                pos_lick[i] = pos_licks.iloc[0]

            if m<.5:
                if rewards[i]>0 and max_pos[i]>rzone1[1]:
                    pcnt[i] = 0
                elif max_pos[i]<rzone1[1]:
                    pcnt[i]=1
            elif m>.5:
                if rewards[i]>0:
                    pcnt[i] = 1
                elif max_pos[i]<rzone1[0]:
                    pcnt[i] = 0
            elif m == .5:
                if zone0_licks[i]>0:
                    pcnt[i] = 0
                elif zone1_licks[i]>0:
                    pcnt[i]=1
    trial_info = {'morphs':morphs,'max_pos':max_pos,'rewards':rewards,'zone0_licks':zone0_licks,'zone1_licks':zone1_licks,'zone0_speed':zone0_speed,
                 'zone1_speed':zone1_speed,'pcnt':pcnt,'wallJitter':wallJitter,'towerJitter':towerJitter,'bckgndJitter':bckgndJitter,'clickOn':clickOn,
                 'pos_lick':pos_lick}
    return trial_info, tstart_inds, teleport_inds


def avg_by_morph(morphs,mat):
    ''''''
    morphs_u = np.unique(morphs)
    ndim = len(mat.shape)
    if ndim==1:
        pcnt_mean = np.zeros([morphs_u.shape[0],])
    elif ndim==2:
        pcnt_mean = np.zeros([morphs_u.shape[0],mat.shape[1]])
    else:
        raise(Exception("mat is wrong number of dimensions"))

    for i,m in enumerate(morphs_u):
        if ndim==1:
            pcnt_mean[i] = np.nanmean(mat[morphs==m])
        if ndim ==2:
            pcnt_mean[i,:] = np.nanmean(mat[morphs==m,:])
    return np.squeeze(pcnt_mean)




def smooth_raster(x,mat,ax=None,smooth=False,sig=2,vals=None):
    '''plot mat ( ntrials x len(x)) as a smoothed histogram'''
    if ax is None:
        f,ax = plt.subplots

    if smooth:
        k = Gaussian1DKernel(5)
        for i in range(mat.shape[0]):
            mat[i,:] = convolve(mat[i,:],k,boundary='extend')

    for ind,i in enumerate(np.arange(mat.shape[0]-1,0,-1)):
        if vals is not None:
            ax.fill_between(x,mat[ind,:]+i,y2=i,color=plt.cm.cool(np.float(vals[ind])),linewidth=.001)
        else:
            ax.fill_between(x,mat[ind,:]+i,y2=i,color = 'black',linewidth=.001)
    #ax.set_y
    ax.set_yticks(np.arange(0,mat.shape[0],10))
    ax.set_yticklabels(["%d" % l for l in np.arange(mat.shape[0],0,-10).tolist()])

    return ax

def lick_plot(d,bin_edges,rzone0=(250.,315),rzone1=(350,415),smooth=True,ratio = True):
    '''standard plot for licking behavior'''
    f = plt.figure(figsize=[15,15])

    gs = gridspec.GridSpec(5,5)


    ax = f.add_subplot(gs[0:-1,0:-1])
    ax.axvspan(rzone0[0],rzone0[1],alpha=.2,color=plt.cm.cool(np.float(0)),zorder=0)
    ax.axvspan(rzone1[0],rzone1[1],alpha=.2,color=plt.cm.cool(np.float(1)),zorder=0)
    ax = smooth_raster(bin_edges[:-1],d['all'],vals=d['labels'],ax=ax,smooth=smooth)
    ax.set_ylabel('Trial',size='xx-large')


    meanlr_ax = f.add_subplot(gs[-1,:-1])
    meanlr_ax.axvspan(rzone0[0],rzone0[1],alpha=.2,color=plt.cm.cool(np.float(0)),zorder=0)
    meanlr_ax.axvspan(rzone1[0],rzone1[1],alpha=.2,color=plt.cm.cool(np.float(1)),zorder=0)
    for i, m in enumerate(np.unique(d['labels'])):
        meanlr_ax.plot(bin_edges[:-1],np.nanmean(d[m],axis=0),color=plt.cm.cool(np.float(m)))
    meanlr_ax.set_ylabel('Licks/sec',size='xx-large')
    meanlr_ax.set_xlabel('Position (cm)',size='xx-large')


    if ratio:
        lickrat_ax = f.add_subplot(gs[:-1,-1])
        bin_edges = np.array(bin_edges)
        rzone0_inds = np.where((bin_edges[:-1]>=rzone0[0]) & (bin_edges[:-1] <= rzone0[1]))[0]
        rzone1_inds = np.where((bin_edges[:-1]>=rzone1[0]) & (bin_edges[:-1] <= rzone1[1]))[0]
        rzone_lick_ratio = {}
        for i,m in enumerate(np.unique(d['labels'])):
            zone0_lick_rate = d[m][:,rzone0_inds].mean(axis=1)
            zone1_lick_rate = d[m][:,rzone1_inds].mean(axis=1)
            rzone_lick_ratio[m] = np.divide(zone0_lick_rate,zone0_lick_rate+zone1_lick_rate)
            rzone_lick_ratio[m][np.isinf(rzone_lick_ratio[m])]=np.nan

        for i,m in enumerate(np.unique(d['labels'])):

            trial_index = d['labels'].shape[0] - d['indices'][m]
            lickrat_ax.scatter(rzone_lick_ratio[m],trial_index,
                               c=plt.cm.cool(np.float(m)),s=10)
            k = Gaussian1DKernel(5)
            lickrat_ax.plot(convolve(rzone_lick_ratio[m],k,boundary='extend'),trial_index,c=plt.cm.cool(np.float(m)))
        lickrat_ax.set_yticklabels([])
        lickrat_ax.set_xlabel(r'$\frac{zone_0}{zone_0 + zone_1}  $',size='xx-large')


        for axis in [ax, meanlr_ax, lickrat_ax]:
            for edge in ['top','right']:
                axis.spines[edge].set_visible(False)

        return f, (ax, meanlr_ax, lickrat_ax)
    else:
        for axis in [ax, meanlr_ax]:
            for edge in ['top','right']:
                axis.spines[edge].set_visible(False)

        return f, (ax, meanlr_ax)

def plot_speed(x,d,vals,ax=None,f=None,rzone0=(250,315),rzone1=(350,415)):
    '''plot individual trial and average speed as a function of position along the Track
    x = position, d=dictionary output of by_trial_dict'''
    if ax is None:
        f, ax = plt.subplots(1,2,figsize=[10,5])
    for i,m in enumerate(np.unique(vals)):
        for j in range(d[m].shape[0]):
            tmp = ax[0].plot(x,d[m][j,:],color = plt.cm.cool(np.float(m)),alpha=.1)
        tmp = ax[0].plot(x,np.nanmean(d[m],axis=0),color=plt.cm.cool(np.float(m)),zorder=1)
        tmp = ax[1].plot(x,np.nanmean(d[m],axis=0),color=plt.cm.cool(np.float(m)))

    ax[0].axvspan(rzone0[0],rzone0[1],alpha=.2,color=plt.cm.cool(np.float(0)),zorder=0)
    ax[0].axvspan(rzone1[0],rzone1[1],alpha=.2,color=plt.cm.cool(np.float(1)),zorder=0)
    ax[1].axvspan(rzone0[0],rzone0[1],alpha=.2,color=plt.cm.cool(np.float(0)),zorder=0)
    ax[1].axvspan(rzone1[0],rzone1[1],alpha=.2,color=plt.cm.cool(np.float(1)),zorder=0)
    for edge in ['top','right']:
        ax[0].spines[edge].set_visible(False)
        ax[1].spines[edge].set_visible(False)

    ax[0].set_xlabel('Position')
    ax[0].set_ylabel('Speed cm/s')
    ax[0].set_ylim([-20, 100])
    ax[1].set_ylim([0, 100])
    return f,ax
