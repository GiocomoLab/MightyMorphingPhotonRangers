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
    def align_to_ca(self,info,nplanes=1):
        sess = self.sessions
        '''align behavioral data to timeseries grid of calcium data'''
        # sbx data
        #info = sbx_loadmat(info_file)
        #get number of frames to drop from beginning
        numVRFrames = info['frame'].size
        caInds = [int(i/nplanes) for i in info['frame']]

        origVRData, (rewardedTrials, errorTrials, omissionTrials) = self._interpolate_data(sess)



        numCaFrames = caInds[-1]-caInds[0]+1
        fr = info['resfreq']/info['recordsPerBuffer']

        #posDat = posDat[-numVRFrames:]
        # make new version of this looping through keys
        for key in origVRData.keys():
            if key not in set(['teleport inds', 'tstart inds']):
                origVRData[key] = origVRData[key][-numVRFrames:]


        gridData = {}
        for key in origVRData.keys():
            gridData[key] = np.zeros([numCaFrames])
            gridData[key][:] = np.nan
        gridData['ca_inds'] = np.arange(caInds[0],caInds[-1]+1,1)
        gridData['time'] = np.arange(0,fr*len(caInds))

        gridData['teleport inds'], gridData['tstart inds'] = [],[]
        for rawInd in list(np.unique(caInds)): # find indices that are within time window
            final_ind = rawInd-caInds[0]
            inds_to_avg = np.where(caInds==rawInd)[0]

            gridData['position'][final_ind] = origVRData['position'][inds_to_avg].mean()
            gridData['speed'][final_ind] = origVRData['speed'][inds_to_avg].mean()

            gridData['licks'][final_ind] = origVRData['licks'][inds_to_avg].sum()
            gridData['rewards'][final_ind] =origVRData['rewards'][inds_to_avg].sum()
            gridData['morph'][final_ind] = sp.stats.mode(origVRData['morph'][inds_to_avg],axis=None)[:]
            gridData['teleports'][final_ind] =  origVRData['teleports'][inds_to_avg].sum()
            gridData['tstart'][final_ind] = origVRData['tstart'][inds_to_avg].sum()
            gridData['error lick'][final_ind] = origVRData['error lick'][inds_to_avg].sum()
            gridData['error mask'][final_ind] = sp.stats.mode(origVRData['error mask'][inds_to_avg],axis=None)[:]
            gridData['omission mask'][final_ind] = sp.stats.modd(origVRData['omission mask'][inds_to_avg],axis=None)[:]

            if origVRData['teleports'][inds_to_avg].sum() >0:
                gridData['teleport inds'].append(final_ind)

            if origVRData['tstart'][inds_to_avg].sum() >0:
                gridData['tstart inds'].append(final_ind)


        return gridData, (rewardedTrials, errorTrials, omissionTrials)

    def _interpolate_data(self):
        '''interpolate all behavioral timeseries to 30 Hz common grid...
        for now just converting data to dictionaries'''
        sess = self.sessions
        if isinstance(sess,list):
            lastTime = 0
            for i in range(len(sess)):
                if i == 0:
                    # lick file
                    lickDat = np.genfromtxt(self.basestr + sess[i] + "_Licks.txt",dtype='float',delimiter='\t')
                    # c_1  r realtimeSinceStartup

                    posDat = np.genfromtxt(self.basestr + sess[i] + "_Pos.txt",dtype = 'float', delimiter='\t')
                    # pos.z realtimeSinceStartup morph true_delta_z

                    lastTime = posDat[-1:1]
                else:
                    tmpLickDat = np.genfromtxt(self.basestr + sess[i] + "_Licks.txt",dtype='float',delimiter='\t')
                    tmpLickDat[:,2] = tmpLickDat[:,2]+lastTime

                    tmpPosDat = np.genfromtxt(self.basestr + sess[i] + "_Pos.txt",dtype = 'float', delimiter='\t')
                    tmpPosDat[:,1] = posDat[:,1]+lastTime

                    lickDat = np.vstack((lickDat,tmpLickDat))
                    posDat = np.vstack((posDat,tmpPosDat))

                    lastTime = tmpPosDat[-1:1]
        else:
            # lick file
            lickDat = np.genfromtxt(self.basestr + sess + "_Licks.txt",dtype='float',delimiter='\t')
            # c_1  r realtimeSinceStartup

            # reward file
            rewardDat = np.genfromtxt(self.basestr + sess + "_Rewards.txt",dtype='float',delimiter='\t')

            # timeout collision files
            toDat = np.genfromtxt(self.basestr + sess + "_Timeout.txt",dtype='float',delimiter='\t')

            posDat = np.genfromtxt(self.basestr + sess + "_Pos.txt",dtype = 'float', delimiter='\t')
            # pos.z realtimeSinceStartup morph true_delta_z

        if lickDat.shape[0] != posDat.shape[0]:
            print("lick data and position data not of consistent lengths. deal with this!!!!")

        gridData = {}
        gridData['time'] = posDat[:,1]
        gridData['position'] = posDat[:,0]
        gridData['speed'] = self._calc_speed(posDat[:,3],posDat[:,1])
        gridData['licks'] = lickDat[:,0]
        gridData['morph'] = posDat[:,2]
        gridData['rewards'] = lickDat[:,1]

        # find teleport and tstart_inds before resampling to prevent errors
        tstart_inds_vec,teleport_inds_vec = np.zeros([posDat.shape[0],]), np.zeros([posDat.shape[0],])
        teleport_inds = np.where(np.ediff1d(posDat[:,0])<=-250)[0]

        tstart_inds = np.append([0],teleport_inds[:-1])
        for ind in range(tstart_inds.shape[0]):  # for teleports
            while (posDat[tstart_inds[ind],0]<0) : # while position is negative
                if tstart_inds[ind] < posDat.shape[0]-1: # if you haven't exceeded the vector length
                    tstart_inds[ind]=tstart_inds[ind]+ 1 # go up one index
                else: # otherwise you should be the last teleport and delete this index
                    print("deleting last index from trial start")
                    tstart_inds=np.delete(tstart_inds,ind)
                    break

        tstart_inds_vec = np.zeros([posDat.shape[0],])
        tstart_inds_vec[tstart_inds] = 1

        teleport_inds_vec = np.zeros([posDat.shape[0],])
        teleport_inds_vec[teleport_inds] = 1

        gridData['teleport inds'] = teleport_inds
        gridData['tstart inds'] = tstart_inds


        gridData['teleports'] = teleport_inds_vec
        gridData['tstart'] = tstart_inds_vec


        gridData['error lick'] = np.zeros([posDat.shape[0],])
        gridData['error mask'] = np.zeros([posDat.shape[0],]) # 1 if error
        gridData['omission mask'] = np.zeros([posDat.shape[0],]) # 1 if omission

        # makes sure last frame is a teleport
        if teleport_inds.shape[0]<tstart_inds.shape[0]:
            teleport_inds = np.append(teleport_inds,posDat.shape[0]-1)
            gridData['teleports'][-1]=1

        errorTrials, rewardedTrials, omissionTrials, morphList = [], [], [], []
        for trial in range(tstart_inds.shape[0]):
            if np.max(gridData['position'][tstart_inds[trial]:teleport_inds[trial]]) < 445:
                gridData['error lick'][np.argmax(gridData['position'][tstart_inds[trial]:teleport_inds[trial]])+tstart_inds[trial]] = 1
                errorTrials.append(trial)
                gridData['error mask'][tstart_inds[trial]:teleport_inds[trial]] = 1

            elif np.max(gridData['rewards'][tstart_inds[trial]:teleport_inds[trial]])>0 and \
            np.max(gridData['position'][tstart_inds[trial]:teleport_inds[trial]]) >= 445:
                rewardedTrials.append(trial)
            else:
                omissionTrials.append(trial)
                gridData['omission mask'][tstart_inds[trial]:teleport_inds[trial]] = 1

            morphList.append(gridData['morph'][tstart_inds[trial]:teleport_inds[trial]].max())

        # self.origVRData =gridData
        # self.rewardedTrials = rewardedTrials
        # self.errorTrials = errorTrials
        # self.omissionTrials = omissionTrials
        # self.morphList = morphList
        return gridData, (rewardedTrials, errorTrials, omissionTrials, morphList)


    def make_trial_matrices(self,gridData):
        ntrials = len(gridData['tstart inds'])
        bin_edges = np.arange(0,450,10).tolist()
        bin_centers = np.arange(5,445,5)

        print(ntrials)
        print(len(gridData['tstart inds']))
        print(len(gridData['teleport inds']))
        trial_matrices = {}
        for key in ['speed','licks','rewards']:
            trial_matrices[key] = np.zeros([ntrials,len(bin_edges)-1])

        for trial in range(ntrials):
            for key in ['speed','licks','rewards']:
                firstI, lastI = gridData['tstart inds'][trial], gridData['teleport inds'][trial]
                if key == 'speed':
                    map, occ = self._rate_map(gridData[key][firstI:lastI],gridData['position'][firstI:lastI],accumulate='mean')
                    trial_matrices[key][trial,:] = map
                else:
                    map, occ = self._rate_map(gridData[key][firstI:lastI],gridData['position'][firstI:lastI],accumulate='sum')
                    trial_matrices[key][trial,:] = map
        # self.trial_matrices = trial_matrices
        return trial_matrices, bin_edges, bin_centers



    def _rate_map(self,vec,position,bin_edges=np.arange(0,450,10).tolist(),accumulate='mean'):
        #bin_edges = np.arange(min_pos,max_pos,bin_size).tolist()
        frmap = np.zeros([len(bin_edges)-1,])
        occupancy = np.zeros([len(bin_edges)-1,])
        for i, (edge1,edge2) in enumerate(zip(bin_edges[:-1],bin_edges[1:])):
            if np.where((position>edge1) & (position<=edge2))[0].shape[0]>0:
                if accumulate == 'mean':
                    frmap[i] = vec[(position>edge1) & (position<=edge2)].mean()
                elif accumulate == 'sum':
                    frmap[i] = vec[(position>edge1) & (position<=edge2)].sum()

                occupancy[i] = np.where((position>edge1) & (position<=edge2))[0].shape[0]
            else:
                pass
        return frmap, occupancy/occupancy.ravel().sum()


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



    def _calc_speed(self,pos,t, dx = True, toSmooth = True ):
        '''calculate speed from position and time vectors'''
        dt = np.ediff1d(t,to_end=1)
        dt[dt==0.] = np.nan
        if dx:
            rawSpeed = np.divide(pos,t)
        else:
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


    # def save_sessions(self,overwrite=False):
    #     '''if session file does not exist, make it'''
    #
    #     for sess in self.sessions:
    #         fname = self.basestr + sess + '.json'
    #         try:
    #             open(fname,'r')
    #             if overwrite:
    #                 os.remove(fname)
    #                 self._save_single_session(sess)
    #         except:
    #             self._save_single_session(sess)
    #
    #
    #
    # def load_sessions(self,verbose = False):
    #     '''if session file exists, load it. If not, create it'''
    #     D, R = {}, {}
    #     for sess in self.sessions:
    #         if verbose:
    #             print('loading ' + sess)
    #         fname = self.basestr + sess + '.json'
    #         try:
    #             #print(fname)
    #             D[sess], R[sess] = self._load_single_session(fname)
    #         except:
    #             self._save_single_session(sess)
    #             D[sess], R[sess] = self._load_single_session(fname)
    #
    #     self.gridData = D
    #     self.rewardTrig = R
    #     return R, D
    #
    #
    # def concatenate_sessions(self,sessions=[]):
    #     '''concatenate numpy arrays of data'''
    #     if len(sessions) == 0:
    #         sessions = self.sessions
    #
    #     Dall, Rall = {}, {}
    #
    #     for key in self.rewardTrig[sessions[0]].keys():
    #         if key == 'time':
    #             Rall[key] = self.rewardTrig[sessions[0]][key]
    #         else:
    #             Rall[key] = np.concatenate(tuple([self.rewardTrig[sess][key] for sess in sessions]),axis = 0)
    #
    #     for key in self.gridData[sessions[0]].keys():
    #         Dall[key] = np.concatenate(tuple([self.gridData[sess][key] for sess in sessions]))
    #
    #     return Rall, Dall
    #
    #
    # def _save_single_session(self,sess):
    #     gridData = self._interpolate_data(sess) # interpolate onto single grid
    #     #dataDict = self._find_single_trials(gridData) # make lists of lists for trials
    #     rewardData = self._reward_trig_dat(gridData) # create numpy arrays
    #
    #     # save json
    #     fname = self.basestr + sess + '.json'
    #     with open(fname,'w') as f:
    #         json.dump({'time grid':self._make_jsonable(gridData),'reward trig':self._make_jsonable(rewardData)},f)
    #
    # def _make_jsonable(self,obj):
    #     '''convert all numpy arrays to lists or lists of lists to save as JSON files'''
    #     obj['tolist'] = []
    #     for key in obj.keys():
    #         if isinstance(obj[key],np.ndarray):
    #             obj[key] = obj[key].tolist()
    #             obj['tolist'].append(key)
    #     return obj
    #
    # def _load_single_session(self,filename):
    #     '''load json file and return all former numpy arrays to such objects'''
    #     with open(filename) as f:
    #         d = json.load(f) # return saved instance
    #
    #     return self._unjson(d['time grid']), self._unjson(d['reward trig'])
    #
    # def _unjson(self,obj):
    #     '''convert lists to numpy arrays'''
    #     for key in obj['tolist']:
    #         obj[key] = np.array(obj[key])
    #     return obj
