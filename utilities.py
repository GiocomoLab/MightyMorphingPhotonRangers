import numpy as np
import h5py
import scipy as sp
from random import randrange
from scipy.ndimage.filters import gaussian_filter


def load_ca_mat(fname):
    ca_dat = {}
    try:
        with h5py.File(fname,'r') as f:
            for k,v in f.items():
                try:
                    ca_dat[k] = np.array(v)
                except:
                    print(k + "not made into numpy array")
                    ca_dat[k]=v
    except:
        ca_dat = sp.io.loadmat(fname)
        for key in ca_dat.keys():
            if isinstance(ca_dat[key],np.ndarray):
                ca_dat[key] = ca_dat[key].T
    return ca_dat


def spatial_info(frmap,occupancy):
    '''calculate spatial information'''
    ncells = frmap.shape[1]

    SI = []
    #p_map = np.zeros(frmap.shape)
    for i in range(ncells):
        p_map = gaussian_filter(frmap[:,i],2)/frmap[:,i].sum()
        #p_map = np.squeeze(frmap[:,i]/frmap[:,i].sum())
        denom = np.multiply(p_map,occupancy).sum()

        si = 0
        for c in range(frmap.shape[0]):
            if (p_map[c]<0) or (occupancy[c]<0):
                print("we have a problem")
            if (p_map[c] != 0) and (occupancy[c]!=0):
                si+= occupancy[c]*p_map[c]*np.log2(p_map[c]/denom)

        SI.append(si)

    return np.array(SI)

def rate_map(C,position,bin_size=10,min_pos = 0, max_pos=350):

    bin_edges = np.arange(min_pos,max_pos,bin_size).tolist()
    frmap = np.zeros([len(bin_edges)-1,C.shape[1]])
    occupancy = np.zeros([len(bin_edges)-1,])
    for i, (edge1,edge2) in enumerate(zip(bin_edges[:-1],bin_edges[1:])):
        if np.where((position>edge1) & (position<=edge2))[0].shape[0]>0:
            frmap[i] = C[(position>edge1) & (position<=edge2),:].mean(axis=0)
            occupancy[i] = np.where((position>edge1) & (position<=edge2))[0].shape[0]
        else:
            pass
    return frmap, occupancy/occupancy.ravel().sum()

def spatial_info_perm_test(SI,C,position,nperms = 10000):

    if len(C.shape)>2:
        C = np.expand_dims(C,1)

    shuffled_SI = np.zeros([nperms,C.shape[1]])

    for perm in range(nperms):
        pos_perm = np.roll(position,randrange(position.shape[0]))
        fr,occ = rate_map(C,pos_perm)
        shuffled_SI[perm,:] = spatial_info(fr,occ)

    p = np.zeros([C.shape[1],])
    for cell in range(C.shape[1]):
        p[cell] = np.where(SI[cell]>shuffled_SI[:,cell])[0].shape[0]/nperms

    return p


def cnmf_com(A,d1,d2,d3):
    pass
    # return centers

def trial_tensor(C,labels,trig_inds,pre=50,post=50):
    '''create a tensor of trial x time x neural dimension'''

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
