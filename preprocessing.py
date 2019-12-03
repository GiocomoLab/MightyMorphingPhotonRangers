import numpy as np
import h5py
import scipy as sp
import scipy.stats
import scipy.io as spio
import scipy.interpolate
import scipy.signal
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
import h5py
import utilities as u

def loadmat_sbx(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    info = _check_keys(data)['info']
    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2; factor = 1
    elif info['channels'] == 2:
        info['nChan'] = 1; factor = 2
    elif info['channels'] == 3:
        info['nChan'] = 1; factor = 2

     # Determine number of frames in whole file
    info['max_idx'] = int(os.path.getsize(filename[:-4] + '.sbx')/info['recordsPerBuffer']/info['sz'][1]*factor/4-1)
    info['fr'] = info['resfreq']/info['recordsPerBuffer']
    return info

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''

    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''

    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def load_ca_mat(f):

    try:
        ca_dat = spio.loadmat(f, struct_as_record=False, squeeze_me=True)
    except:
        # ca_dat = {}
        with h5py.File(f,'r') as h:
            ca_dat = {k:np.array(v) for k,v in h.items() if k in {'C','C_keep','C_dec','S','S_dec'} }
    return ca_dat

def load_scan_sess(sess,analysis='s2p',plane=0,fneu_coeff=.7):
    VRDat = behavior_dataframe(sess['data file'],scanmats=sess['scanmat'],concat=False)

    # load imaging
    info = loadmat_sbx(sess['scanmat'])
    if analysis == "cnmf":
        ca_dat = load_ca_mat(sess['scanfile'])
        try:
            C = ca_dat['C'][1:,:]#[info['frame'][0]-1:info['frame'][-1]]
        except:
            C = ca_dat['C_keep'][1:,:]#[info['frame'][0]-1:info['frame'][-1]]

        Cd = ca_dat['C_dec']#[info['frame'][0]-1:info['frame'][-1]]
        S = ca_dat['S_dec']#[info['frame'][0]-1:info['frame'][-1]]
        frame_diff = VRDat.shape[0]-C.shape[0]
        print('frame diff',frame_diff)
        assert (frame_diff==0), "something is wrong with aligning VR and calcium data"

        return VRDat,C,S,None

    elif analysis == "s2p":

        folder = os.path.join(sess['s2pfolder'],'plane%i' % plane)

        F= np.load(os.path.join(folder,'F.npy'))
        Fneu = np.load(os.path.join(folder,'Fneu.npy'))
        iscell =  np.load(os.path.join(folder,'iscell.npy'))
        S = np.load(os.path.join(folder,'spks.npy'))
        F_ = F-fneu_coeff*Fneu
        C=F_[iscell[:,0]>0,:].T
        C = u.df(C)
        S=S[iscell[:,0 ]>0,:].T
        frame_diff = VRDat.shape[0]-C.shape[0]
        print('frame diff',frame_diff)
        assert (frame_diff==0), "something is wrong with aligning VR and calcium data"
        return VRDat,C,S,F_[iscell[:,0]>0,:].T
    else:
        return

def load_session_db(dir = "G:\\My Drive\\"):
    '''open the sessions sqlite database and add some columns'''

    vr_fname = os.path.join(dir,"VR_Data","TwoTower","behavior.sqlite")
    print(vr_fname)
    conn = sql.connect(vr_fname)
    df = pd.read_sql("SELECT MouseName, DateFolder, SessionNumber,Track, RewardCount, Imaging, ImagingRegion FROM sessions",conn)
    sdir = os.path.join(dir,"VR_Data","TwoTower")
    df['DateTime'] = [datetime.strptime(s,'%d_%m_%Y') for s in df['DateFolder']]
    df['data file'] = [ build_VR_filename(df['MouseName'].iloc[i],
                                           df['DateFolder'].iloc[i],
                                           df['Track'].iloc[i],
                                           df['SessionNumber'].iloc[i],serverDir=sdir) for i in range(df.shape[0])]
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
    # add s2p filefolder
    df['s2pfolder']=[build_s2p_folder(df.iloc[i],serverDir=twop_dir) for i in range(df.shape[0])]

    conn.close()
    return df

def build_s2p_folder(df,serverDir="G:\\My Drive\\2P_Data\\TwoTower\\"):

    res_folder = os.path.join(serverDir,df['MouseName'],df['DateFolder'],df['Track'],"%s_*%s_*" % (df['Track'],df['SessionNumber']),'suite2p')
    match= glob(res_folder)
    assert len(match)<2, "multiple matching subfolders"
    if len(match)<1:
        return None
    else:
        return match[0]



def build_2P_filename(mouse,date,scene,sess,serverDir = "G:\\My Drive\\2P_Data\\TwoTower\\"):
    ''' use sessions database inputs to build appropriate filenames for 2P data'''

    results_fname = os.path.join(serverDir,mouse,date,scene,"%s_*%s_*_cnmf_results.mat" % (scene,sess))
    results_file=glob(results_fname)
    if len(results_file)==0:
        results_fname = os.path.join(serverDir,mouse,date,scene,"%s_*%s_*_cnmf_results_pre.mat" % (scene,sess))
        results_file=glob(results_fname)

    info_fname = os.path.join(serverDir,mouse,date,scene,"%s_*%s_*[0-9].mat" % (scene,sess))
    info_file = glob(info_fname)

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
        print("file doesn't exist, errors to come!!!")
        #raise Exception("file doesn't exist")


def _VR_align_to_2P(vr_dframe,infofile, n_imaging_planes = 1):
    '''align behavior to 2P sample times using splines'''

    info = loadmat_sbx(infofile)
    fr = info['fr'] # frame rate
    lr = fr*512. # line rate

    ## on Feb 6, 2019 noticed that Alex Attinger's new National Instruments board
    ## created a floating ground on my TTL circuit. This caused a bunch of extra TTLs
    ## due to unexpected grounding of the signal.


    orig_ttl_times = info['frame']/fr + info['line']/lr # including error ttls
    dt_ttl = np.diff(np.insert(orig_ttl_times,0,0)) # insert zero at beginning and calculate delta ttl time
    tmp = np.zeros(dt_ttl.shape)
    tmp[dt_ttl<.005] = 1 # find ttls faster than 200 Hz (unrealistically fast - probably a ttl which bounced to ground)
    # ensured outside of this script that this finds the true start ttl on every scan
    mask = np.insert(np.diff(tmp),0,0) # find first ttl in string that were too fast
    mask[mask<0] = 0
    print('num aberrant ttls',tmp.sum())

    frames = info['frame'][mask==0] # should be the original ttls up to a 1 VR frame error
    lines = info['line'][mask==0]

    ttl_times = frames/fr + lines/lr
    numVRFrames = frames.shape[0]

    ca_df = pd.DataFrame(columns = vr_dframe.columns, index = np.arange(info['max_idx']))
    ca_time = np.arange(0,1/fr*info['max_idx'],1/fr)
    if (ca_time.shape[0]-ca_df.shape[0])==1:
        print('one frame correction')
        ca_time = ca_time[:-1] #np.append(ca_time,ca_time[-1]+1/fr)
    print(info['max_idx'],ca_time.shape,ca_df.shape,numVRFrames)

    ca_df.loc[:,'time'] = ca_time
    mask = ca_time>=ttl_times[0]

    vr_dframe = vr_dframe.iloc[-numVRFrames:]
    print(ttl_times.shape,vr_dframe.shape)
    f_mean = sp.interpolate.interp1d(ttl_times,vr_dframe['pos']._values,axis=0,kind='slinear')
    # f_mean = sp.interpolate.interp1d(vr_time,vr_dframe['pos']._values,axis=0,kind='slinear')
    ca_df.loc[mask,'pos'] = f_mean(ca_time[mask])
    ca_df.loc[~mask,'pos']=-500.

    near_list = ['morph','clickOn','towerJitter','wallJitter','bckgndJitter']
    f_nearest = sp.interpolate.interp1d(ttl_times,vr_dframe[near_list]._values,axis=0,kind='nearest')
    # f_nearest = sp.interpolate.interp1d(vr_time,vr_dframe[near_list]._values,axis=0,kind='nearest')
    ca_df.loc[mask,near_list] = f_nearest(ca_time[mask])
    ca_df.fillna(method='ffill',inplace=True)
    ca_df.loc[~mask,near_list]=-1.

    cumsum_list = ['dz','lick','reward','tstart','teleport','rzone']

    f_cumsum = sp.interpolate.interp1d(ttl_times,np.cumsum(vr_dframe[cumsum_list]._values,axis=0),axis=0,kind='slinear')
    # f_cumsum = sp.interpolate.interp1d(vr_time,np.cumsum(vr_dframe[cumsum_list]._values,axis=0),axis=0,kind='slinear')
    ca_cumsum = np.round(np.insert(f_cumsum(ca_time[mask]),0,[0,0, 0,0 ,0,0],axis=0))
    #print('cumsum',ca_cumsum[-1,:])
    if ca_cumsum[-1,-2]<ca_cumsum[-1,-3]:
        ca_cumsum[-1,-2]+=1


    ca_df.loc[mask,cumsum_list] = np.diff(ca_cumsum,axis=0)
    ca_df.loc[~mask,cumsum_list] = 0.

    # fill na here
    ca_df.loc[np.isnan(ca_df['teleport']._values),'teleport']=0
    ca_df.loc[np.isnan(ca_df['tstart']._values),'tstart']=0


    k = Gaussian1DKernel(5)
    cum_dz = convolve(np.cumsum(ca_df['dz']._values),k,boundary='extend')
    ca_df['dz'] = np.ediff1d(cum_dz,to_end=0)


    ca_df['speed'].interpolate(method='linear',inplace=True)
    ca_df['speed']=np.array(np.divide(ca_df['dz'],np.ediff1d(ca_df['time'],to_begin=1./fr)))
    ca_df['speed'].iloc[0]=0


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

    cumsum_list = ['dz','lick','reward','tstart','teleport','rzone']

    f_cumsum = sp.interpolate.interp1d(vr_time,np.cumsum(frame[cumsum_list]._values,axis=0),axis=0,kind='slinear')
    ca_cumsum = np.round(np.insert(f_cumsum(ca_time),0,[0,0, 0 ,0,0,0],axis=0))
    if ca_cumsum[-1,-1]<ca_cumsum[-1,-2]:
        ca_cumsum[-1,-1]+=1


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

    frame = pd.read_sql('''SELECT * FROM data''',sess_conn)
    k = Gaussian1DKernel(5)
    frame['speed']=np.array(np.divide(frame['dz'],np.ediff1d(frame['time'],to_begin=.001)))
    frame['lick rate'] = np.array(np.divide(frame['lick'],np.ediff1d(frame['time'],to_begin=.001)))

    if fix_teleports:
        tstart_inds_vec,teleport_inds_vec = np.zeros([frame.shape[0],]), np.zeros([frame.shape[0],])
        pos = frame['pos']._values
        pos[pos<-50] = -50
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
        # print('fix teleports',frame.shape,tstart_inds.shape,teleport_inds.shape)
        tstart_inds_vec[tstart_inds] = 1

        teleport_inds_vec = np.zeros([frame.shape[0],])
        teleport_inds_vec[teleport_inds] = 1
        #print('fix teleports post sub',teleport_inds_vec.sum(),tstart_inds_vec.sum())
        frame['teleport']=teleport_inds_vec
        frame['tstart']=tstart_inds_vec
        #print('frame fill',frame.teleport.sum(),frame.tstart.sum())

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
