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

def loadmat_sbx(filename):
    """
    this function should be called instead of direct spio.loadmat

    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    print(filename)
    data_ = sp.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data_)


def _check_keys(dict):
    """
    checks if entries in dictionary rare mat-objects. If yes todict is called to change them to nested dictionaries
    """

    for key in dict:
        if isinstance(dict[key], sp.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])

    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """

    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sp.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def load_ca_mat(fname):
    """load results from cnmf"""

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

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def rate_map(C,position,bin_size=10,min_pos = 0, max_pos=450):

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

def make_pos_bin_trial_matrices(arr, pos, tstart, tstop,method = 'mean',bin_size=5):
    ntrials = np.sum(tstart)
    bin_edges = np.arange(0,450+bin_size,bin_size)
    bin_centers = bin_edges[:-1]+bin_size/2
    bin_edges = bin_edges.tolist()
    #print(len(bin_edges),bin_centers.shape)


    if len(arr.shape)<2:
        arr = np.expand_dims(arr,axis=1)

    trial_mat = np.zeros([int(ntrials),len(bin_edges)-1,arr.shape[1]])
    trial_mat[:] = np.nan
    tstart_inds, tstop_inds = np.where(tstart==1)[0],np.where(tstop==1)[0]
    for trial in range(int(ntrials)):

            firstI, lastI = tstart_inds[trial], tstop_inds[trial]
            #print(arr[firstI:lastI])
            map, occ = rate_map(arr[firstI:lastI],pos[firstI:lastI],bin_size=bin_size)
            #nans,x = nan_helper(map)
            #map[nans] = np.interp(x(nans),x(~nans),map[~nans])
            trial_mat[trial,:,:] = map
            #print(map.ravel())
    # self.trial_matrices = trial_matrices
    return np.squeeze(trial_mat), bin_edges, bin_centers

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



def build_2P_filename(mouse,date,scene,sess,serverDir = "G:\\My Drive\\2P_Data\\TwoTower"):
    results_fname = os.path.join(serverDir,mouse,date,scene,"%s_*%s_*_cnmf_results_pre.mat" % (scene,sess))
    results_file=glob(results_fname)
    info_fname = os.path.join(serverDir,mouse,date,scene,"%s_*%s_*.mat" % (scene,sess))
    info_file = glob(info_fname)
    #results_file=glob("%s\\%s\\%s\\%s\\%s_*%s_*_cnmf_results_pre.mat" % (serverDir, mouse, date, scene, scene, sess))
    #info_file = glob("%s\\%s\\%s\\%s\\%s_*%s_*.mat" % (serverDir, mouse, date, scene, scene, sess))

    if len(info_file)==0:
        #raise Exception("file doesn't exist")
        return None, None
    elif len(info_file)>0 and len(results_file)==0:
        return None, info_file[0]
    else:
        return results_file[0], info_file[0]

def build_VR_filename(mouse,date,scene,session,serverDir = "G:\\My Drive\\VR_Data\\TwoTower"):
    fname = os.path.join(serverDir,mouse,date,"%s_%s.sqlite" % (scene,session))
    #file=glob("%s\\%s\\%s\\%s_%s.sqlite" % (serverDir, mouse, date, scene, session))
    file=glob(fname)
    if len(file)==1:
        return file[0]
    else:
        print("%s\\%s\\%s\\%s_%s.sqlite" % (serverDir, mouse, date, scene, session))
        raise Exception("file doesn't exist")

def trial_type_dict(mat,type_vec):
    d = {'all': np.squeeze(mat)}
    d['labels'] = type_vec
    d['indices']={}
    for i,m in enumerate(np.unique(type_vec)):
        d['indices'][m] = np.where(type_vec==m)[0]
        d[m] = d['all'][d['indices'][m],:]

    return d

def _VR_align_to_2P(frame,infofile, n_imaging_planes = 1):

    info = loadmat_sbx(infofile)['info']
    numVRFrames = info['frame'].size
    caInds = np.array([int(i/n_imaging_planes) for i in info['frame']])

    numCaFrames = caInds[-1]-caInds[0]+1
    fr = info['resfreq']/info['recordsPerBuffer']

    frame = frame.iloc[-numVRFrames:]
    frame['ca inds'] = caInds
    #tmp_frame = frame.groupby(['ca inds'])
    ca_df = pd.DataFrame(columns = frame.columns,index=np.arange(numCaFrames))

    ca_df['time'] = np.arange(0,1/fr*numCaFrames,1/fr)

    vr_time = frame['time']._values
    vr_time = vr_time - vr_time[0]

    ca_time = np.arange(0,np.min([ca_df['time'].iloc[-1], vr_time[-1]]),1/fr)
    underhang = int(np.round((1/fr*numCaFrames-ca_time[-1])*fr))

    print(ca_df.iloc[:-underhang+1].shape)
    #f_mean = sp.interpolate.interp1d(vr_time,frame[['pos','dz']]._values,axis=0,kind='slinear')
    f_mean = sp.interpolate.interp1d(vr_time,frame['pos']._values,axis=0,kind='slinear')
    print(f_mean(ca_time).shape)
    ca_df.loc[ca_df.time<=vr_time[-1],'pos'] = f_mean(ca_time)

    near_list = ['morph','clickOn','towerJitter','wallJitter','bckgndJitter']
    f_nearest = sp.interpolate.interp1d(vr_time,frame[near_list]._values,axis=0,kind='nearest')
    ca_df.loc[ca_df.time<=vr_time[-1],near_list] = f_nearest(ca_time)

    cumsum_list = ['dz','lick','reward','tstart','teleport']

    f_cumsum = sp.interpolate.interp1d(vr_time,np.cumsum(frame[cumsum_list]._values,axis=0),axis=0,kind='slinear')
    ca_cumsum = np.round(np.insert(f_cumsum(ca_time),0,[0,0, 0 ,0,0],axis=0))
    if ca_cumsum[-1,-1]<ca_cumsum[-1,-2]:
        ca_cumsum[-1,-1]+=1
    #ca_df[cumsum_list].iloc[1:-underhang+1]=np.diff(ca_cumsum,axis=0

    ca_df.loc[ca_df.time<=vr_time[-1],cumsum_list] = np.diff(ca_cumsum,axis=0)

    ca_df.fillna(method='ffill',inplace=True)
    k = Gaussian1DKernel(5)
    cum_dz = convolve(np.cumsum(ca_df['dz']._values),k,boundary='extend')
    ca_df['dz'] = np.ediff1d(cum_dz,to_end=0)


    ca_df['speed'].interpolate(method='linear',inplace=True)
    ca_df['speed']=np.array(np.divide(ca_df['dz'],np.ediff1d(ca_df['time'],to_begin=1./fr)))
    ca_df['speed'].iloc[0]=0

    # ca_df['speed'] = convolve(ca_df['speed']._values,k,boundary='extend')
    ca_df['lick rate'] = np.array(np.divide(ca_df['lick'],np.ediff1d(ca_df['time'],to_begin=1./fr)))
    ca_df['lick rate'] = convolve(ca_df['lick rate']._values,k,boundary='extend')
    ca_df[['reward','tstart','teleport','lick','clickOn','towerJitter','wallJitter','bckgndJitter']].fillna(value=0,inplace=True)

    return ca_df

def _VR_interp(frame):

    fr = 30

    vr_time = frame['time']._values
    vr_time = vr_time - vr_time[0]
    ca_time = np.arange(0,vr_time[-1],1/fr)
    ca_df = pd.DataFrame(columns = frame.columns,index=np.arange(ca_time.shape[0]))

    ca_df['time'] = ca_time

    f_mean = sp.interpolate.interp1d(vr_time,frame['pos']._values,axis=0,kind='slinear')
    ca_df['pos'] = f_mean(ca_time)

    near_list = ['morph','clickOn','towerJitter','wallJitter','bckgndJitter']
    f_nearest = sp.interpolate.interp1d(vr_time,frame[near_list]._values,axis=0,kind='nearest')
    ca_df[near_list] = f_nearest(ca_time)

    cumsum_list = ['dz','lick','reward','tstart','teleport']

    f_cumsum = sp.interpolate.interp1d(vr_time,np.cumsum(frame[cumsum_list]._values,axis=0),axis=0,kind='slinear')
    ca_cumsum = np.round(np.insert(f_cumsum(ca_time),0,[0,0, 0 ,0,0],axis=0))
    if ca_cumsum[-1,-1]<ca_cumsum[-1,-2]:
        ca_cumsum[-1,-1]+=1
    #ca_df[cumsum_list].iloc[1:-underhang+1]=np.diff(ca_cumsum,axis=0

    ca_df[cumsum_list] = np.diff(ca_cumsum,axis=0)

    ca_df.fillna(method='ffill',inplace=True)
    k = Gaussian1DKernel(5)
    cum_dz = convolve(np.cumsum(ca_df['dz']._values),k,boundary='extend')
    ca_df['dz'] = np.ediff1d(cum_dz,to_end=0)


    ca_df['speed'].interpolate(method='linear',inplace=True)
    ca_df['speed']=np.array(np.divide(ca_df['dz'],np.ediff1d(ca_df['time'],to_begin=1./fr)))
    ca_df['speed'].iloc[0]=0

    # ca_df['speed'] = convolve(ca_df['speed']._values,k,boundary='extend')
    ca_df['lick rate'] = np.array(np.divide(ca_df['lick'],np.ediff1d(ca_df['time'],to_begin=1./fr)))
    ca_df['lick rate'] = convolve(ca_df['lick rate']._values,k,boundary='extend')
    ca_df[['reward','tstart','teleport','lick','clickOn','towerJitter','wallJitter','bckgndJitter']].fillna(value=0,inplace=True)

    return ca_df

def _get_frame(f,fix_teleports=True):
    sess_conn = sql.connect(f)
    frame = pd.read_sql('''SELECT time, pos, dz, morph, lick, reward, tstart, teleport, clickOn, towerJitter
                , wallJitter, bckgndJitter FROM data''',sess_conn)

    #frame.loc[frame.dz>.2,'dz']=.2
    k = Gaussian1DKernel(5)
    frame['speed']=np.array(np.divide(frame['dz'],np.ediff1d(frame['time'],to_begin=.001)))
    frame['lick rate'] = np.array(np.divide(frame['lick'],np.ediff1d(frame['time'],to_begin=.001)))

    if fix_teleports:
        tstart_inds_vec,teleport_inds_vec = np.zeros([frame.shape[0],]), np.zeros([frame.shape[0],])
        pos = frame['pos']._values
        teleport_inds = np.where(np.ediff1d(pos,to_end=0)<=-50)[0]
        tstart_inds = np.append([0],teleport_inds[:-1]+1)
        for ind in range(tstart_inds.shape[0]):  # for teleports
            while (pos[tstart_inds[ind]]<0) or (pos[tstart_inds[ind]]>5) : # while position is negative
                if tstart_inds[ind] < pos.shape[0]-1: # if you haven't exceeded the vector length
                    tstart_inds[ind]=tstart_inds[ind]+ 1 # go up one index
                else: # otherwise you should be the last teleport and delete this index
                    print("deleting last index from trial start")
                    tstart_inds=np.delete(tstart_inds,ind)
                    break

        tstart_inds_vec = np.zeros([frame.shape[0],])
        tstart_inds_vec[tstart_inds] = 1

        teleport_inds_vec = np.zeros([frame.shape[0],])
        teleport_inds_vec[teleport_inds] = 1
        frame['teleport']=teleport_inds_vec
        frame['tstart']=tstart_inds_vec


    return frame



def behavior_dataframe(filenames,scanmats=None,concat = True, sig=10):

    if scanmats is None:
        if isinstance(filenames,list):
            frames = [_VR_interp(_get_frame(f)) for f in filenames]
            df = pd.concat(frames,ignore_index=True)
        else:
            df = _VR_interp(_get_frame(filenames))
        df['trial number'] = np.cumsum(df['teleport'])

        if isinstance(filenames,list):
            if concat:
                return df
            else:
                return frames
        else:
            return df

    else:
        if isinstance(filenames,list):
            if len(filenames)!=len(scanmats):
                raise Exception("behavior and scanfile lists must be of the same length")
            else:
                frames = [_VR_align_to_2P(_get_frame(f),s) for (f,s) in zip(filenames,scanmats)]
                df = pd.concat(frames,ignore_index=True)
        else:

            df = _VR_align_to_2P(_get_frame(filenames),scanmats)
        df['trial number'] = np.cumsum(df['teleport'])

        if isinstance(filenames,list):
            if concat:

                return df
            else:
                return frames
        else:
            return df
# percent correct
def by_trial_info(data,rzone0=(250,315),rzone1=(350,415)):
    tstart_inds, teleport_inds = data.index[data.tstart==1],data.index[data.teleport==1]
    #print(tstart_inds.shape[0],teleport_inds.shape[0])
    trial_info={}
    morphs = np.zeros([tstart_inds.shape[0],])
    max_pos = np.zeros([tstart_inds.shape[0]])
    rewards = np.zeros([tstart_inds.shape[0]])
    zone0_licks = np.zeros([tstart_inds.shape[0]])
    zone1_licks = np.zeros([tstart_inds.shape[0]])
    zone0_speed = np.zeros([tstart_inds.shape[0]])
    zone1_speed = np.zeros([tstart_inds.shape[0]])
    pcnt = np.zeros([tstart_inds.shape[0]]); pcnt[:] = np.nan
    wallJitter= np.zeros([tstart_inds.shape[0]])
    towerJitter= np.zeros([tstart_inds.shape[0]])
    bckgndJitter= np.zeros([tstart_inds.shape[0]])
    clickOn= np.zeros([tstart_inds.shape[0]])
    for (i,(s,f)) in enumerate(zip(tstart_inds,teleport_inds)):
        sub_frame = data[s:f]
        m, counts = sp.stats.mode(sub_frame['morph'],nan_policy='omit')
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
                 'zone1_speed':zone1_speed,'pcnt':pcnt,'wallJitter':wallJitter,'towerJitter':towerJitter,'bckgndJitter':bckgndJitter,'clickOn':clickOn}
    return trial_info

def avg_by_morph(morphs,pcnt):
    morphs_u = np.unique(morphs)
    pcnt_mean = np.zeros([morphs_u.shape[0]])
    for i,m in enumerate(morphs_u):
        pcnt_mean[i] = np.nanmean(pcnt[morphs==m])
    return pcnt_mean





def load_session_db(dir = "G:\\My Drive\\VR_Data\\TwoTower"):
    fname = os.path.join(dir,"behavior.sqlite")
    conn = sql.connect(fname)
    df = pd.read_sql("SELECT MouseName, DateFolder, SessionNumber,Track, RewardCount, Imaging FROM sessions",conn)
    df['DateTime'] = [datetime.strptime(s,'%d_%m_%Y') for s in df['DateFolder']]
    df['data file'] = [ build_VR_filename(df['MouseName'].iloc[i],
                                           df['DateFolder'].iloc[i],
                                           df['Track'].iloc[i],
                                           df['SessionNumber'].iloc[i],serverDir=dir) for i in range(df.shape[0])]
    choose_first, choose_second = lambda x: x[0], lambda x: x[1]
    df['scanfile'] = [choose_first(build_2P_filename(df['MouseName'].iloc[i],
                                        df['DateFolder'].iloc[i],
                                        df['Track'].iloc[i],
                                        df['SessionNumber'].iloc[i],serverDir=dir)) for i in range(df.shape[0])]
    df['scanmat'] = [choose_second(build_2P_filename(df['MouseName'].iloc[i],
                                        df['DateFolder'].iloc[i],
                                        df['Track'].iloc[i],
                                        df['SessionNumber'].iloc[i],serverDir=dir)) for i in range(df.shape[0])]
    conn.close()
    return df

def smooth_raster(x,mat,ax=None,smooth=False,sig=2,vals=None):
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
    ax[0].set_ylim([0, 100])
    ax[1].set_ylim([0, 100])
    return f,ax
