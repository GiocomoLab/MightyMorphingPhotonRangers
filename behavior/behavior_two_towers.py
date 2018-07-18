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
        self.basestr = "Z:\\VR\\TwoTower\\" + mouse + "\\"
    def align_to_ca(self,info,nplanes=1):
        sess = self.sessions
        '''align behavioral data to timeseries grid of calcium data'''
        # sbx data
        #info = sbx_loadmat(info_file)
        #get number of frames to drop from beginning
        numVRFrames = info['frame'].size
        caInds = [int(i/nplanes) for i in info['frame']]

        origVRData, (rewardedTrials, errorTrials, omissionTrials, morphList) = self._interpolate_data()



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

            if len(inds_to_avg)>0:
                gridData['position'][final_ind] = origVRData['position'][inds_to_avg].mean()
                gridData['speed'][final_ind] = origVRData['speed'][inds_to_avg].mean()

                gridData['licks'][final_ind] = origVRData['licks'][inds_to_avg].sum()
                gridData['lick rate'][final_ind] = origVRData['lick rate'][inds_to_avg].mean()
                gridData['rewards'][final_ind] =origVRData['rewards'][inds_to_avg].sum()

                gridData['morph'][final_ind] = origVRData['morph'][inds_to_avg].max()
                gridData['teleports'][final_ind] =  origVRData['teleports'][inds_to_avg].sum()
                gridData['tstart'][final_ind] = origVRData['tstart'][inds_to_avg].sum()
                gridData['error lick'][final_ind] = origVRData['error lick'][inds_to_avg].sum()
                gridData['error mask'][final_ind] = origVRData['error mask'][inds_to_avg].max()
                gridData['omission mask'][final_ind] = origVRData['omission mask'][inds_to_avg].max()

                if origVRData['teleports'][inds_to_avg].sum() >0:
                    gridData['teleport inds'].append(final_ind)

                if origVRData['tstart'][inds_to_avg].sum() >0:
                    gridData['tstart inds'].append(final_ind)
            else:
                gridData['position'][final_ind] = gridData['position'][final_ind-1]
                gridData['speed'][final_ind] = gridData['speed'][final_ind-1]

                gridData['licks'][final_ind] = 0
                gridData['lick rate'][final_ind] = gridData['lick rate'][final_ind-1]
                gridData['rewards'][final_ind] =0

                gridData['morph'][final_ind] = gridData['morph'][final_ind-1]
                gridData['teleports'][final_ind] =  0
                gridData['tstart'][final_ind] = 0
                gridData['error lick'][final_ind] = 0
                gridData['error mask'][final_ind] = gridData['error mask'][final_ind-1]
                gridData['omission mask'][final_ind] = gridData['omission mask'][final_ind-1]


        for final_ind in range(numCaFrames):
            if np.isnan(gridData['position'][final_ind]):
                gridData['position'][final_ind] = gridData['position'][final_ind-1]
                gridData['speed'][final_ind] = gridData['speed'][final_ind-1]

                gridData['licks'][final_ind] = 0
                gridData['lick rate'][final_ind] = gridData['lick rate'][final_ind-1]
                gridData['rewards'][final_ind] =0

                gridData['morph'][final_ind] = gridData['morph'][final_ind-1]
                gridData['teleports'][final_ind] =  0
                gridData['tstart'][final_ind] = 0
                gridData['error lick'][final_ind] = 0
                gridData['error mask'][final_ind] = gridData['error mask'][final_ind-1]
                gridData['omission mask'][final_ind] = gridData['omission mask'][final_ind-1]


        return gridData, (rewardedTrials, errorTrials, omissionTrials,morphList)

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

                    lastTime = posDat[-1,1]
                else:
                    tmpLickDat = np.genfromtxt(self.basestr + sess[i] + "_Licks.txt",dtype='float',delimiter='\t')
                    tmpLickDat[:,2] = tmpLickDat[:,2]+lastTime

                    tmpPosDat = np.genfromtxt(self.basestr + sess[i] + "_Pos.txt",dtype = 'float', delimiter='\t')
                    tmpPosDat[:,1] = tmpPosDat[:,1]+lastTime

                    lickDat = np.vstack((lickDat,tmpLickDat))
                    posDat = np.vstack((posDat,tmpPosDat))

                    lastTime = tmpPosDat[-1,1]
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
        gridData['delta z'] = posDat[:,3]
        gridData['speed'] = np.zeros(gridData['position'].shape)
        gridData['licks'] = lickDat[:,0]
        gridData['lick rate'] = self._calc_speed(lickDat[:,0],posDat[:,1],dx=True)
        gridData['morph'] = posDat[:,2]
        gridData['rewards'] = lickDat[:,1]

        # find teleport and tstart_inds before resampling to prevent errors
        tstart_inds_vec,teleport_inds_vec = np.zeros([posDat.shape[0],]), np.zeros([posDat.shape[0],])
        teleport_inds = np.where(np.ediff1d(posDat[:,0])<=-50)[0]
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

        gridData['tower jitter'] = np.zeros([posDat.shape[0],])
        gridData['wall jitter'] = np.zeros([posDat.shape[0],])
        gridData['background jitter'] = np.zeros([posDat.shape[0],])
        gridData['trial number'] = np.zeros([posDat.shape[0],])

        # makes sure last frame is a teleport
        if teleport_inds.shape[0]<tstart_inds.shape[0]:
            teleport_inds = np.append(teleport_inds,posDat.shape[0]-1)
            gridData['teleports'][-1]=1

        errorTrials, rewardedTrials, omissionTrials, morphList = [], [], [], []

        for trial in range(tstart_inds.shape[0]):

            if np.max(gridData['position'][tstart_inds[trial]:teleport_inds[trial]]) < 425:
                gridData['error lick'][np.argmax(gridData['position'][tstart_inds[trial]:teleport_inds[trial]])+tstart_inds[trial]] = 1
                errorTrials.append(trial)
                gridData['error mask'][tstart_inds[trial]:teleport_inds[trial]] = 1

            elif np.max(gridData['rewards'][tstart_inds[trial]:teleport_inds[trial]])>0 and \
            np.max(gridData['position'][tstart_inds[trial]:teleport_inds[trial]]) >= 425:
                rewardedTrials.append(trial)
            else:
                omissionTrials.append(trial)
                gridData['omission mask'][tstart_inds[trial]:teleport_inds[trial]] = 1

            morphList.append(gridData['morph'][tstart_inds[trial]:teleport_inds[trial]].max())

            gridData['speed'][tstart_inds[trial]:teleport_inds[trial]]= self._calc_speed(posDat[tstart_inds[trial]:teleport_inds[trial],0],posDat[tstart_inds[trial]:teleport_inds[trial],1],dx=False)


        gridData['speed'][np.where(gridData['speed']<-10)[0]]=0
        # self.origVRData =gridData
        # self.rewardedTrials = rewardedTrials
        # self.errorTrials = errorTrials
        # self.omissionTrials = omissionTrials
        # self.morphList = morphList
        return gridData, (rewardedTrials, errorTrials, omissionTrials, morphList)

    def to_sql_dicts(self):
        '''interpolate all behavioral timeseries to 30 Hz common grid...
        for now just converting data to dictionaries'''
        sess = self.sessions

        # lick file
        try:
            lickDat = np.genfromtxt(self.basestr + sess + "_Licks.txt",dtype='float',delimiter='\t')
        # c_1  r realtimeSinceStartup
        except:
            lickDat = np.genfromtxt(self.basestr + sess + "Licks.txt",dtype='float',delimiter='\t')

        # reward file
        try:
            rewardDat = np.genfromtxt(self.basestr + sess + "_Rewards.txt",delimiter='\t')
        except:
            rewardDat = np.genfromtxt(self.basestr + sess + "Rewards.txt",delimiter='\t')

        # timeout collision files
        try:
            rewardDat = np.vstack((rewardDat,np.genfromtxt(self.basestr + sess + "_Timeout.txt",dtype=None,delimiter='\t')))
        except:
            try:
                rewardDat = np.vstack((rewardDat,np.genfromtxt(self.basestr + sess + "Timeout.txt",dtype=None,delimiter='\t')))
            except:
                pass

        try:
            posDat = np.genfromtxt(self.basestr + sess + "_Pos.txt",dtype = 'float', delimiter='\t')
        except:
            posDat = np.genfromtxt(self.basestr + sess + "Pos.txt",dtype = 'float', delimiter='\t')
        # pos.z realtimeSinceStartup morph true_delta_z
        try:
            manRewardDat = np.reshape(np.genfromtxt(self.basestr + sess + "ManRewards.txt", delimiter='\t'),[-1,2])
        except:
            manRewardDat = np.array([])
        if lickDat.shape[0] != posDat.shape[0]:
            print("lick data and position data not of consistent lengths. deal with this!!!!")

        gridData = {}
        gridData['time'] = posDat[:,1]
        gridData['position'] = posDat[:,0]
        gridData['delta z'] = posDat[:,3]
        gridData['speed'] = np.zeros(gridData['position'].shape)
        gridData['licks'] = lickDat[:,0]
        gridData['lick rate'] = self._calc_speed(lickDat[:,0],posDat[:,1],dx=True)
        gridData['morph'] = posDat[:,2]
        gridData['rewards'] = lickDat[:,1]

        # find teleport and tstart_inds before resampling to prevent errors
        tstart_inds_vec,teleport_inds_vec = np.zeros([posDat.shape[0],]), np.zeros([posDat.shape[0],])
        teleport_inds = np.where(np.ediff1d(posDat[:,0])<=-50)[0]
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

        gridData['tower jitter'] = np.zeros([posDat.shape[0],])
        gridData['wall jitter'] = np.zeros([posDat.shape[0],])
        gridData['background jitter'] = np.zeros([posDat.shape[0],])
        gridData['click on'] = np.zeros([posDat.shape[0],])
        if(rewardDat.shape[1]==7):
            for r in range(rewardDat.shape[0]):
                rInd = np.argmin(np.abs(gridData['time']-rewardDat[r,1]))
                gridData['click on'][rInd] = int(rewardDat[r,3]==True)
                gridData['tower jitter'][rInd] = rewardDat[r,4]
                gridData['wall jitter'][rInd] = rewardDat[r,5]
                gridData['background jitter'][rInd] = rewardDat[r,6]



        # makes sure last frame is a teleport
        if teleport_inds.shape[0]<tstart_inds.shape[0]:
            teleport_inds = np.append(teleport_inds,posDat.shape[0]-1)
            gridData['teleports'][-1]=1


        for trial in range(tstart_inds.shape[0]):

            for key in ['click on', 'tower jitter', 'wall jitter', 'background jitter']:
                tmp_dat = gridData[key][tstart_inds[trial]:teleport_inds[trial]]
                if tmp_dat.shape[0] != 0:
                    val = np.argmax(np.abs(tmp_dat))
                    gridData[key][tstart_inds[trial]:teleport_inds[trial]] = tmp_dat[val]


        gridData['man rewards'] = np.zeros([posDat.shape[0],])
        print(manRewardDat.shape)
        if manRewardDat.shape[0]>0:
            for row in range(manRewardDat.shape[0]):
                mInd = np.argmin(np.abs(gridData['time']-manRewardDat[row,0]))
                gridData['man rewards'][mInd] = 1

        return gridData


    def make_trial_matrices(self,gridData):
        ntrials = len(gridData['tstart inds'])
        bin_edges = np.arange(0,450,5).tolist()
        bin_centers = np.arange(2.5,447.5,5)


        trial_matrices = {}
        for key in ['speed','licks','rewards','lick rate']:
            trial_matrices[key] = np.zeros([ntrials,len(bin_edges)-1])

        for trial in range(ntrials):
            for key in ['speed','licks','rewards', 'lick rate']:
                firstI, lastI = gridData['tstart inds'][trial], gridData['teleport inds'][trial]
                if key in ['speed', 'lick rate']:
                    map, occ = self._rate_map(gridData[key][firstI:lastI],gridData['position'][firstI:lastI],accumulate='mean')
                    trial_matrices[key][trial,:] = map

                else:
                    map, occ = self._rate_map(gridData[key][firstI:lastI],gridData['position'][firstI:lastI],accumulate='sum')
                    trial_matrices[key][trial,:] = map
        # self.trial_matrices = trial_matrices
        return trial_matrices, bin_edges, bin_centers



    def _rate_map(self,vec,position,bin_edges=np.arange(0,450,5).tolist(),accumulate='mean'):
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
