import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob as glb
import scipy as sp
import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter
import json
import astropy.convolution as astconv
import pandas as pd

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



class process_data:

    def __init__(self,mouse,sessions,workdir=""):
        self.sessions = sessions
        self.mouse = mouse
        self.data = {}
        self.basestr = "Z:\\VR\\2AFC_V3\\" + mouse + "\\"


    def save_sessions(self,overwrite=False):
        '''if session file does not exist, make it'''

        for sess in self.sessions:
            fname = self.basestr + sess + '.json'
            try:
                open(fname,'r')
                if overwrite:
                    os.remove(fname)
                    self._save_single_session(sess)
            except:
                self._save_single_session(sess)



    def load_sessions(self,verbose = False):
        '''if session file exists, load it. If not, create it'''
        D, R = {}, {}
        for sess in self.sessions:
            if verbose:
                print('loading ' + sess)
            fname = self.basestr + sess + '.json'
            try:
                #print(fname)
                D[sess], R[sess] = self._load_single_session(fname)
            except:
                self._save_single_session(sess)
                D[sess], R[sess] = self._load_single_session(fname)

        self.gridData = D
        self.rewardTrig = R
        return R, D


    def concatenate_sessions(self,sessions=[]):
        '''concatenate numpy arrays of data'''
        if len(sessions) == 0:
            sessions = self.sessions

        Dall, Rall = {}, {}

        for key in self.rewardTrig[sessions[0]].keys():
            if key == 'time':
                Rall[key] = self.rewardTrig[sessions[0]][key]
            else:
                Rall[key] = np.concatenate(tuple([self.rewardTrig[sess][key] for sess in sessions]),axis = 0)

        for key in self.gridData[sessions[0]].keys():
            Dall[key] = np.concatenate(tuple([self.gridData[sess][key] for sess in sessions]))

        return Rall, Dall


    def _save_single_session(self,sess):
        gridData = self._interpolate_data(sess) # interpolate onto single grid
        #dataDict = self._find_single_trials(gridData) # make lists of lists for trials
        rewardData = self._reward_trig_dat(gridData) # create numpy arrays

        # save json
        fname = self.basestr + sess + '.json'
        with open(fname,'w') as f:
            json.dump({'time grid':self._make_jsonable(gridData),'reward trig':self._make_jsonable(rewardData)},f)

    def _make_jsonable(self,obj):
        '''convert all numpy arrays to lists or lists of lists to save as JSON files'''
        obj['tolist'] = []
        for key in obj.keys():
            if isinstance(obj[key],np.ndarray):
                obj[key] = obj[key].tolist()
                obj['tolist'].append(key)
        return obj

    def _load_single_session(self,filename):
        '''load json file and return all former numpy arrays to such objects'''
        with open(filename) as f:
            d = json.load(f) # return saved instance

        return self._unjson(d['time grid']), self._unjson(d['reward trig'])

    def _unjson(self,obj):
        '''convert lists to numpy arrays'''
        for key in obj['tolist']:
            obj[key] = np.array(obj[key])
        return obj

    def align_to_ca(self,sess,info,save_data =True,nplanes=1):
        '''align behavioral data to timeseries grid of calcium data'''
        # sbx data
        #info = sbx_loadmat(info_file)
        #get number of frames to drop from beginning
        numVRFrames = info['frame'].size
        caInds = [int(i/nplanes) for i in info['frame']]


        lickDat = np.genfromtxt(self.basestr + sess + "_Licks.txt",dtype='float',delimiter='\t')
        lickDat = lickDat[-numVRFrames:,:]


        # reward file
        rewardDat = np.genfromtxt(self.basestr + sess + "_Rewards.txt",dtype='float',delimiter='\t')
        posDat = np.genfromtxt(self.basestr + sess + "_Pos.txt",dtype = 'float', delimiter='\t')
        posDat = posDat[-numVRFrames:]
        speed = self._calc_speed(posDat[:,0],posDat[:,1])
        # pos.z realtimeSinceStartup

        #  use 2P syncing pulses to make common grid for the data
        numCaFrames = caInds[-1]-caInds[0]+1
        fr = info['resfreq']/info['recordsPerBuffer']
        #gridData = np.zeros([tGrid.size,10])
        gridData = {}
        gridData['position'] = np.empty([numCaFrames,])
        gridData['position'][:] = np.nan
        gridData['speed'] = np.zeros([numCaFrames,])
        gridData['speed'][:] = np.nan
        gridData['licks'] = np.zeros([numCaFrames,])
        gridData['licks'][:] = np.nan
        gridData['morph'] = np.zeros([numCaFrames,])
        gridData['rewards'] = np.zeros([numCaFrames,])
        gridData['ca_inds'] = np.arange(caInds[0],caInds[-1]+1,1)

        timeSinceStartup = np.zeros([numCaFrames,])
        timeSinceStartup[:] =np.nan
        nonNan = []

        # find tstart_inds before resampling to prevent errors
        tstart_inds_vec = np.zeros([numCaFrames,])
        tstart_inds_raw = np.where(np.ediff1d(posDat[:,0],to_begin = -900)<=-200)[0]
        print(tstart_inds_raw.shape)
        if tstart_inds_raw.shape[0]>rewardDat.shape[0]:
            for ind in range(tstart_inds_raw.shape[0]-1): # skip last
                while (posDat[tstart_inds_raw[ind],0]<0) :
                    tstart_inds_raw[ind]=tstart_inds_raw[ind]+1
        else:
            for ind in range(tstart_inds_raw.shape[0]): # skip last
                while (posDat[tstart_inds_raw[ind],0]<0) :
                    tstart_inds_raw[ind]=tstart_inds_raw[ind]+1
        tstart_inds_raw[-1] = posDat.shape[0]-1
        tstart_inds_vec_raw = np.zeros([posDat.shape[0],])
        tstart_inds_vec_raw[tstart_inds_raw] = 1

        #tstart_inds = []
        #tstart_counter = 0
        for rawInd in list(np.unique(caInds)): # find indices that are within time window
            final_ind = rawInd-caInds[0]
            inds_to_avg = np.where(caInds==rawInd)[0]

            gridData['position'][final_ind] = np.nanmean(posDat[inds_to_avg,0])
            gridData['speed'][final_ind] = speed[inds_to_avg].mean()

            gridData['licks'][final_ind] = lickDat[inds_to_avg,0].sum()
            gridData['rewards'][final_ind] =lickDat[inds_to_avg,1].max()
            timeSinceStartup[final_ind] = lickDat[inds_to_avg,2].mean()




            tstart_inds_vec[final_ind] = tstart_inds_vec_raw[inds_to_avg].max()

        tstart_inds = np.where(np.diff(tstart_inds_vec)>0)[0]

        # make morph nan during inter-trial interval
        # interpolate nans
        gridData = pd.DataFrame.from_dict(gridData)
        gridData.interpolate(method='nearest',inplace=True)



        print(tstart_inds.shape)
        print(rewardDat.shape)
        if tstart_inds.shape[0]<=rewardDat.shape[0]:
            tstart_inds = np.append(tstart_inds,caInds[-1])
            print("quit mid trial")



        reward_inds= []
        for trial in range(rewardDat.shape[0]):
        #print(rewardDat[trial,:])

            # make morph vector same length as everything else

            gridData.iloc[tstart_inds[trial]:tstart_inds[trial+1],gridData.columns.get_loc('morph')]= rewardDat[trial,2]

            # make 'side' vector zeros except for start of reward
            rewardTime = rewardDat[trial,1]
            #print(np.abs(timeSinceStartup-rewardTime)[0:10])
            reward_ind = np.nanargmin(np.abs(timeSinceStartup-rewardTime))
            reward_inds.append(reward_ind)
            #print(timeSinceStartup)
            #print(rewardTime, reward_ind)
            if twoPort:
                gridData.iloc[reward_ind,gridData.columns.get_loc('side')] = rewardDat[trial,3]

            rval = np.max(gridData['rewards'].values[tstart_inds[trial]:tstart_inds[trial+1]])
            rval_ind = np.argmax(gridData['rewards'].values[tstart_inds[trial]:tstart_inds[trial+1]])

            #rval = np.max(gridData.iloc[tstart_inds[trial]:tstart_inds[trial+1],gridData.columns.get_loc('rewards')].values)
            gridData.iloc[tstart_inds[trial]:tstart_inds[trial]+rval_ind,gridData.columns.get_loc('rewards')] = -rval
            gridData.iloc[tstart_inds[trial]+rval_ind:tstart_inds[trial+1],gridData.columns.get_loc('rewards')] = rval

        if save_data:
            pass

        return gridData, tstart_inds[:-1], reward_inds



    def _interpolate_data(self,sess):
        '''interpolate all behavioral timeseries to 30 Hz common grid'''

        # lick file
        lickDat = np.genfromtxt(self.basestr + sess + "_Licks.txt",dtype='float',delimiter='\t')
        # c_1  r realtimeSinceStartup

        # reward file
        rewardDat = np.genfromtxt(self.basestr + sess + "_Rewards.txt",dtype='float',delimiter='\t')

        # timeout collision files
        try:
            toDat = np.genfromtxt(self.basestr + sess + "_Timeout.txt",dtype='float',delimiter='\t')
        except:
            pass

        posDat = np.genfromtxt(self.basestr + sess + "_Pos.txt",dtype = 'float', delimiter='\t')
        # pos.z realtimeSinceStartup morph true_delta_z


        # lick data will have earliest timepoint
        #  use time since startup to make common grid for the data
        dt = 1./30.
        endT = lickDat[-1,2] + dt
        tGrid = np.arange(0,endT,dt)

        #gridData = np.zeros([tGrid.size,10])
        gridData = {}
        gridData['position'] = np.zeros([tGrid.size-1,])
        gridData['position'][:] = np.nan
        gridData['speed'] = np.zeros([tGrid.size-1,])
        gridData['speed'][:] = np.nan
        gridData['licks'] = np.zeros([tGrid.size-1,])
        gridData['licks'][:] = np.nan
        gridData['morph'] = np.zeros([tGrid.size-1,])
        gridData['morph'][:]=np.nan
        gridData['reward collisions'] = np.zeros([tGrid.size-1,])
        gridData['timeout collisions'] = np.zeros([tGrid.size-1])
        gridData['rewards'] = np.zeros([tGrid.size-1,])
        gridData['time'] = tGrid[:-1]
        gridData['trial'] = np.zeros([tGrid.size-1,])
        gridData['teleports'] = np.zeros([tGrid.size-1,])
        gridData['error lick'] = np.zeros([tGrid.size-1,])
        gridData['trial start'] = np.zeros([tGrid.size-1,])
        gridData['error mask'] = np.zeros([tGrid.size -1]) # 1 if error
        gridData['omission mask'] = np.zeros([tGrid.size-1]) # 1 if omission



        # find teleport and tstart_inds before resampling to prevent errors
        tstart_inds_vec,teleport_inds_vec = np.zeros([tGrid.size,]), np.zeros([tGrid.size,])
        teleport_inds_raw = np.where(np.ediff1d(posDat[:,0],to_begin = -900)<=-200)[0]
        tstart_inds_raw = np.copy(teleport_inds_raw)
        print(teleport_inds_raw.shape)



        for ind in range(tstart_inds_raw.shape[0]):  # for teleports
            while (posDat[tstart_inds_raw[ind],0]<0) : # while position is negative
                if tstart_inds_raw[ind] < posDat.shape[0]-1: # if you haven't exceeded the vector length
                    tstart_inds_raw[ind]=tstart_inds_raw[ind]+ 1 # go up one index
                else: # otherwise you should be the last teleport and delete this index
                    print("deleting last index from trial start")
                    tstart_inds_raw=np.delete(tstart_inds_raw,ind)
                    break

        tstart_inds_vec_raw = np.zeros([posDat.shape[0],])
        tstart_inds_vec_raw[tstart_inds_raw] = 1

        teleport_inds_vec_raw = np.zeros([posDat.shape[0],])
        teleport_inds_vec_raw[teleport_inds_raw] = 1


        for i in range(tGrid.size-1): # find indices that are within time window
            if i%100 == 0:
                print(i)
            tWin = tGrid[[i, i+1]]

            lickInd = np.where((lickDat[:,2]>=tWin[0]) & (lickDat[:,2] <= tWin[1]))[0]

            rewardInd = np.where((rewardDat[:,1]>=tWin[0]) & (rewardDat[:,1] <= tWin[1]))[0]

            toInd = np.where((toDat[:,1]>=tWin[0]) & (toDat[:,1] <= tWin[1]))[0]

            posInd = np.where((posDat[:,1]>=tWin[0])  & (posDat[:,1]<= tWin[1]))[0]

            ## build indices of lickDat

            if posInd.size>0:
                # 1) position
                gridData['position'][i] = posDat[posInd,0].mean()

                gridData['speed'][i] = posDat[posInd,3].mean()

                gridData['morph'][i] = posDat[posInd,2].max()


                gridData['teleports'][i] = teleport_inds_vec_raw[posInd].max()
                if teleport_inds_vec_raw[posInd].max() >0 and posDat[posInd,0].mean() < 445.:
                    gridData['error lick'][i] = 1

                gridData['trial start'][i] = tstart_inds_vec_raw[posInd].max()

                tstart_inds_vec[i], teleport_inds_vec[i] = tstart_inds_vec_raw.max(), teleport_inds_vec_raw.max()

            else:

                # 1) position
                gridData['position'][i] = np.nan

                # 2) speed
                gridData['speed'][i] = np.nan

            if lickInd.size>0:

                gridData['licks'][i] = lickDat[lickInd,0].sum()
                gridData['rewards'][i] = lickDat[lickInd,1].sum()


            # 5) reward flag
            if rewardInd.size>0:
                gridData['reward collisions'][i] = rewardDat[rewardInd,3].max()

            if toInd.size>0:
                gridData['timeout collisions'][i] = toDat[toInd,3].max()

            # 6) manual reward flag
            #if mRewardInd.size > 0:
            #    gridData['manual rewards'][i] = mRewardDat[mRewardInd,1].min()

        tstart_inds = np.where(np.diff(tstart_inds_vec)>0)[0]
        teleport_inds = np.where(np.diff(teleport_inds_vec)>0)[0]+1
        # makes sure last frame is a teleport
        if teleport_inds.shape[0]<tstart_inds.shape[0]:
            teleport_inds = np.append(teleport_inds,tGrid.size)

        gridData = pd.DataFrame.from_dict(gridData)
        gridData.interpolate(method='nearest',inplace=True)


        errorTrials, rewardedTrials, omissionTrials = [], [], []
        for trial in range(tstart_inds.shape[0]):
            # if error licks > 1
            # append error trial
            if gridData['error lick'].values[tstart_inds[trial]:teleport_inds[trial]].sum() >1:
                errorTrials.append(trial)

            # else if error licks <1 and rewarded licks > 0
            elif gridData['error lick'].values[tstart_inds[trial]:teleport_inds[trial]].sum() <1 and \
            gridData['rewards'].values[tstart_inds[trial]:teleport_inds[trial]].sum() >1:
            # append rewarded trialStart
                rewardedTrials.append(trial)

            # else if error licks and rewarded licks < 1
            elif gridData['error lick'].values[tstart_inds[trial]:teleport_inds[trial]].sum() <1 and \
            gridData['rewards'].values[tstart_inds[trial]:teleport_inds[trial]].sum() <1:
            # append ommision
                omissionTrials.append(trial)
            else:
                print("trial type not understood")


        return gridData, (tstart_inds,teleport_inds), (rewardedTrials, errorTrials, omissionTrials)


    def _trial_lists(self,gridData,trialStart):
        '''make dictionary for each variable that contains list of np arrays for each trial'''


        #trialLists = [[] for i in range(trialStart.size)]
        trialLists = {}
        trialLists['position'] = []
        trialLists['speed'] = []
        trialLists['licks'] = []
        trialLists['rewards'] = [ ]
        trialLists['time'] = []
        trialLists['morph'] = []

        for i in range(trialStart.size-1):
            trialLists['position'].append(gridData['position'][trialStart[i]:trialStart[i+1]])
            trialLists['speed'].append(gridData['speed'][trialStart[i]:trialStart[i+1]])
            trialLists['licks'].append(gridData['licks'][trialStart[i]:trialStart[i+1]])
            trialLists['rewards'].append(gridData['rewards'][trialStart[i]:trialStart[i+1]])
            trialLists['morph'].append(gridData['morph'][trialStart[i]:trialStart[i+1]].max()*np.ones(gridData['rewards'][trialStart[i]:trialStart[i+1]].shape))
            trialLists['time'].append(gridData['time'][trialStart[i]:trialStart[i+1]])

        return trialLists


    def _reward_zone_responses(self, trialLists):
        pass



    def _calc_speed(self,pos,t, toSmooth = True ):
        '''calculate speed from position and time vectors'''
        dt = np.ediff1d(t,to_end=1)
        dt[dt==0.] = np.nan
        rawSpeed = np.divide(np.ediff1d(pos,to_end=0),dt)

        notTeleports = np.where(np.ediff1d(pos,to_begin=0)>-50)[0]


        inds = [i for i in range(rawSpeed.size)]
        rawSpeed = np.interp(inds,[inds[i] for i in notTeleports],[rawSpeed[i] for i in notTeleports])

        if toSmooth:
            speed = self._smooth_speed(rawSpeed)
        else:
            speed = rawSpeed
        return speed


    def _smooth_speed(self,speed,smooth=10):
        return gaussian_filter(speed,smooth)
