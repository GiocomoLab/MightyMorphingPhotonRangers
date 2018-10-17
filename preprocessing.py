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


def load_ca_mat(fname,fov = [512,796]):
    """load results from cnmf"""

    ca_dat = {}
    try:
        with h5py.File(fname,'r') as f:
            try:
                C = np.array(f['C'])
            except:
                C = np.array(f['C_keep'])

            for k,v in f.items():
                try:
                    if k in ('A_keep', 'A'):
                        ca_dat[k] = sp.sparse.csc_matrix((f[k]['data'],f[k]['ir'],f[k]['jc']),shape=[fov[0]*fov[1],C.shape[1]])
                    else:
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

def load_scan_sess(sess):
    VRDat = behavior_dataframe(sess['data file'],scanmats=sess['scanmat'],concat=False)

    # load imaging
    info = loadmat_sbx(sess['scanmat'])['info']
    ca_dat = load_ca_mat(sess['scanfile'])

    C = ca_dat['C_dec'][info['frame'][0]:info['frame'][-1]+1]
    frame_diff = VRDat.shape[0]-C.shape[0]
    if frame_diff>0:
        VRDat = VRDat.iloc[:-frame_diff]

    if 'A_keep' in ca_dat.keys():
        return VRDat,C, ca_dat['A_keep']
    elif 'A' in ca_dat.keys():
        return VRDat,C, ca_dat['A']

def load_session_db(dir = "G:\\My Drive\\"):
    '''open the sessions sqlite database and add some columns'''

    vr_fname = os.path.join(dir,"VR_Data","TwoTower","behavior.sqlite")
    conn = sql.connect(vr_fname)
    df = pd.read_sql("SELECT MouseName, DateFolder, SessionNumber,Track, RewardCount, Imaging FROM sessions",conn)
    df['DateTime'] = [datetime.strptime(s,'%d_%m_%Y') for s in df['DateFolder']]
    df['data file'] = [ build_VR_filename(df['MouseName'].iloc[i],
                                           df['DateFolder'].iloc[i],
                                           df['Track'].iloc[i],
                                           df['SessionNumber'].iloc[i],serverDir="%s\\VR_Data\\TwoTower\\" % dir) for i in range(df.shape[0])]
    choose_first, choose_second = lambda x: x[0], lambda x: x[1]
    twop_dir = os.path.join(dir,"2P_Data","TwoTower")
    df['scanfile'] = [choose_first(build_2P_filename(df['MouseName'].iloc[i],
                                        df['DateFolder'].iloc[i],
                                        df['Track'].iloc[i],
                                        df['SessionNumber'].iloc[i],serverDir=twop_dir)) for i in range(df.shape[0])]
    df['scanmat'] = [choose_second(build_2P_filename(df['MouseName'].iloc[i],
                                        df['DateFolder'].iloc[i],
                                        df['Track'].iloc[i],
                                        df['SessionNumber'].iloc[i],serverDir=twop_dir)) for i in range(df.shape[0])]
    conn.close()
    return df

def build_2P_filename(mouse,date,scene,sess,serverDir = "G:\\My Drive\\2P_Data\\TwoTower\\"):
    ''' use sessions database inputs to build appropriate filenames for 2P data'''

    results_fname = os.path.join(serverDir,mouse,date,scene,"%s_*%s_*_cnmf_results.mat" % (scene,sess))
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

def build_VR_filename(mouse,date,scene,session,serverDir = "G:\\My Drive\\VR_Data\\TwoTower\\"):
    '''use sessions database to build filenames for behavioral data (also a
    sqlite database)'''
    fname = os.path.join(serverDir,mouse,date,"%s_%s.sqlite" % (scene,session))
    #file=glob("%s\\%s\\%s\\%s_%s.sqlite" % (serverDir, mouse, date, scene, session))
    file=glob(fname)
    if len(file)==1:
        return file[0]
    else:
        print("%s\\%s\\%s\\%s_%s.sqlite" % (serverDir, mouse, date, scene, session))
        raise Exception("file doesn't exist")


def _VR_align_to_2P(frame,infofile, n_imaging_planes = 1):
    '''align behavior to 2P sample times using splines'''

    info = loadmat_sbx(infofile)['info']
    numVRFrames = info['frame'].size
    caInds = np.array([int(i/n_imaging_planes) for i in info['frame']])

    numCaFrames = caInds[-1]-caInds[0]+1
    print(numCaFrames)
    fr = info['resfreq']/info['recordsPerBuffer']

    frame = frame.iloc[-numVRFrames:]
    frame['ca inds'] = caInds
    #tmp_frame = frame.groupby(['ca inds'])
    ca_df = pd.DataFrame(columns = frame.columns,index=np.arange(numCaFrames))
    #print(np.arange(numCaFrames).shape,np.arange(0,1/fr*numCaFrames,1/fr).shape,fr,numCaFrames)
    ca_df['time'] = np.arange(0,1/fr*numCaFrames,1/fr)[:numCaFrames]

    vr_time = frame['time']._values
    vr_time = vr_time - vr_time[0]

    ca_time = np.arange(0,np.min([ca_df['time'].iloc[-1], vr_time[-1]])+.001,1/fr)
    underhang = int(np.round((1/fr*numCaFrames-ca_time[-1])*fr))
    #print(ca_df['time'].iloc[-1],ca_time[-1],vr_time[-1],ca_time.shape)

    #print(ca_df.iloc[:-underhang+1].shape)
    #f_mean = sp.interpolate.interp1d(vr_time,frame[['pos','dz']]._values,axis=0,kind='slinear')
    f_mean = sp.interpolate.interp1d(vr_time,frame['pos']._values,axis=0,kind='slinear')
    #print(ca_time.shape,ca_df.shape)
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
    '''if 2P data doesn't exist interpolates behavioral data onto an even grid'''
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
    '''load a single session's sqlite database for behavior'''
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
    '''loads a list of vr sessions given filenames. capable of concatenating for
    averaging data across sessions'''
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
