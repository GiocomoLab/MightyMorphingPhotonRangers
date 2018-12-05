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
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
import pandas as pd
from datetime import datetime
from glob import glob
import os.path
from astropy.convolution import convolve, Gaussian1DKernel


class LOTrialO:
    def __init__(self,starts,stops,N):
        self.train_mask = np.zeros([N,])
        self.test_mask = np.zeros([N,])
        self.c = 0
        self.starts = starts
        self.stops = stops
        self.N = N

    def next(self):
        if self.c<len(self.starts):
            self.train_mask *= 0
            self.test_mask *= 0
            for t,(start,stop) in enumerate(zip(self.starts,self.stops)):
                if t == self.c:
                    self.test_mask[start:stop]+=1
                else:
                    self.train_mask[start:stop]+=1
            self.c+=1
            return self.train_mask>0,self.test_mask>0
        else:
            return None, None


def make_spline_basis(x,knots=np.arange(-.2,1,.34)):
    '''make cubic spline basis functions'''
    knotfunc = lambda k: np.power(np.multiply(x-k,(x-k)>0),3)
    spline_basis_list = [knotfunc(k) for k in knots.tolist()]
    spline_basis_list += [np.ones(x.shape[0]),x,np.power(x,2)]
    return np.array(spline_basis_list).T

def correct_trial_mask(rewards,starts,stops,N):
    pcnt = np.zeros([N,])
    # get correct trials
    for i,(start,stop) in enumerate(zip(starts,stops)):
        pcnt[start:stop] = int(rewards[i]>0)
    return pcnt


def lick_positions(licks,position):
    lickpos = np.zeros([licks.shape[0],])
    lickpos[:]=np.nan
    lick_inds = np.where(licks>0)[0]
    lickpos[lick_inds]=position[lick_inds]
    return lickpos


def rate_map(C,position,bin_size=5,min_pos = 0, max_pos=450):
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



def make_pos_bin_trial_matrices(arr, pos, tstart, tstop,method = 'mean',bin_size=5,perm=False):
    '''make a ntrials x position x neurons tensor'''
    if tstart.shape[0]>1000: # if binary, leaving in for backwards compatibility
        tstart_inds, tstop_inds = np.where(tstart==1)[0],np.where(tstop==1)[0]
        ntrials = np.sum(tstart)
    else:
        tstart_inds, tstop_inds = tstart, tstop
        ntrials = tstart.shape[0]
    # print('pos bin',tstart_inds.shape,tstop_inds.shape,ntrials)

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


    #print(int(ntrials),tstart_inds.shape,tstop_inds.shape)
    #print(tstart_inds[[0,1,-1]],tstop_inds[[0,1,-1]])
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
    # print('by trial func', tstart_inds.shape,teleport_inds.shape)
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




def smooth_raster(x,mat,ax=None,smooth=False,sig=2,vals=None,tports=None):
    '''plot mat ( ntrials x len(x)) as a smoothed histogram'''
    if ax is None:
        f,ax = plt.subplots

    if smooth:
        k = Gaussian1DKernel(sig)
        for i in range(mat.shape[0]):
            mat[i,:] = convolve(mat[i,:],k,boundary='extend')

    for ind,i in enumerate(np.arange(mat.shape[0]-1,0,-1)):
        if vals is not None:
            ax.fill_between(x,mat[ind,:]+i,y2=i,color=plt.cm.cool(np.float(vals[ind])),linewidth=.001)
        else:
            ax.fill_between(x,mat[ind,:]+i,y2=i,color = 'black',linewidth=.001)

        if tports is not None:
            ax.scatter(tports[ind],i+.5,color=plt.cm.cool(np.float(vals[ind])),marker='x',s=50)
    #ax.set_y
    ax.set_yticks(np.arange(0,mat.shape[0],10))
    ax.set_yticklabels(["%d" % l for l in np.arange(mat.shape[0],0,-10).tolist()])

    return ax
