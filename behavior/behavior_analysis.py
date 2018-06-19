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


def first_half_speed(R,track_length=350.):

    pos,speed = R['position'], R['speed']
    morphVals = np.unique(R['morph'])

    first_half_speed,lickSide = [],[]
    for i in range(morphVals.size):

        first_half_inds = np.where((R['position']>0.)
                        & (R['position']<track_length/2)
                        & (R['morph'] == morphVals[i]))

        first_half_speed.append(speed[first_half_inds])
        lickSide.append(R['licks'][first_half_inds])



    return first_half_speed,lickSide



def reward_trig_lick_rate(R, title = '', plots="all"):



    if plots in set(['all','total']):
        f_all,ax_all = plt.subplots()
        ax_all.plot(R['time'],R['port1 licks'].mean(axis=0),'r',R['time'],R['port2 licks'].mean(axis=0),'k')
        ax_all.set_title('all morphs')
        ax_all.legend(('port1 licks','port2 licks'))
        #f_all.show()


    morphVals = np.unique(R['morph'])
    f_morph, ax_morph = plt.subplots(morphVals.size,1,figsize=(5,5*morphVals.size))
    lLicks_list, rLicks_list = [],[]
    for i in range(morphVals.size):

        plotInds = np.where(R['morph']==morphVals[i])
        #f_morph[i], ax_morph[i] = plt.subplots()
        lLicks, rLicks = R['port1 licks'][plotInds,:], R['port2 licks'][plotInds,:]
        lLicks_list.append(lLicks)
        rLicks_list.append(rLicks)
        if plots in set(['all','morphs']):
            ax_morph[i].plot(R['time'], np.squeeze(lLicks).mean(axis=0),'r',
                            R['time'],np.squeeze(rLicks).mean(axis=0),'k')
            ax_morph[i].set_title((title + ' morph = %f' % morphVals[i] ))
            ax_morph[i].legend(('port1 licks','port2 licks'))

    return lLicks_list,rLicks_list,morphVals, (f_morph, ax_morph)

def cum_wrong(R,window_size = 20):
    morphOrder = R['morph']
    wrongL, wrongR = np.zeros(morphOrder.shape), np.zeros(morphOrder.shape)

    wrongLeftInds = np.where((R['first lick']==1) & (R['morph']== 1))[0]
    wrongRightInds = np.where((R['first lick']==2) & (R['morph']== 0))[0]
    wrongL[wrongLeftInds]=1.
    wrongR[wrongRightInds]=1.



    return np.cumsum(wrongL), np.cumsum(wrongR), np.cumsum(wrongL+wrongR)

def psychometrics(R,title='',plots = 'all'):
    '''plot probability that first lick is port2 given morph value excluding omissions.
    plot percent omissions for each morph value'''

    morphVals = np.unique(R['morph']).tolist()
    pport2 = []
    omissions = []
    lRT, rRT= [], []
    fhandles,axhandles = [],[]

    if 0 not in morphVals:
        morphVals = [0]+morphVals

    if 1 not in morphVals:
        morphVals = morphVals+[1]

    for i in range(len(morphVals)):

        omit = np.where((R['morph']==morphVals[i]) & (R['first lick'] == 0))[0]
        lLick = np.where((R['morph']==morphVals[i]) & (R['first lick'] == 1))[0]
        rLick = np.where((R['morph'] == morphVals[i]) & (R['first lick']==2))[0]

        if lLick.size>0:
            lRT.append(np.array(R['RT'][lLick]))

        else:
            lRT.append(np.array([np.nan]))

        if rLick.size>0:
            rRT.append(np.array(R['RT'][rLick]))
        else:
            rRT.append(np.array([np.nan]))



        if lLick.size+rLick.size>0:
            pport2.append(float(rLick.size)/float(lLick.size+rLick.size))
        else:
            pport2.append(np.nan)

        if omit.size+lLick.size+rLick.size > 0:
            omissions.append(float(omit.size)/float(omit.size+lLick.size+rLick.size))
        else:
            omissions.append(np.nan)


    if ('psych' in plots) or plots=='all' :
        f_pc, ax_pc = plt.subplots()
        ax_pc.plot(morphVals,pport2)
        ax_pc.set_ylabel('P(lick port2)')
        ax_pc.set_ylim([0, 1])
        ax_pc.set_title(title)
        fhandles.append(f_pc)
        axhandles.append(ax_pc)
        #f_pc.show()

    if ('omissions' in plots) or plots =='all':
        f_o, ax_o = plt.subplots()
        ax_o.plot(morphVals,omissions)
        ax_o.set_title(title+' percent omissions')
        ax_o.set_ylim([0, 1])
        fhandles.append(f_o)
        axhandles.append(ax_o)

    rRT_mu = [np.nanmean(rRT[i]) for i in range(len(rRT))]
    lRT_mu = [np.nanmean(lRT[i]) for i in range(len(lRT))]
    combRT_mu = []
    for i in range(len(lRT)):

        if np.isnan(lRT_mu[i]) and not np.isnan(rRT_mu[i]):
            combRT_mu.append(rRT_mu[i])
        elif np.isnan(rRT_mu[i]) and not np.isnan(lRT_mu[i]):
            combRT_mu.append(lRT_mu[i])
        elif np.isnan(lRT_mu[i]) and np.isnan(rRT_mu[i]):
            combRT_mu.append(np.nan)
        else:
            combRT_mu.append(
            (lRT_mu[i]*lRT[i].size + rRT_mu[i]*rRT[i].size)/(lRT[i].size +
            rRT[i].size))

    #combRT_mu = [(lRT[i].sum() + rRT[i].sum())/(lRT[i].size+rRT[i].size) for i in range(len(lRT))]

    if ('RT' in plots) or plots == 'all':
        f_rt, ax_rt = plt.subplots()
        ax_rt.plot(morphVals,combRT_mu, 'b', morphVals,lRT_mu,'k',morphVals,rRT_mu,'r')
        ax_rt.set_title(title+' reaction time')
        ax_rt.legend(('combined','left licks','right licks'))
        fhandles.append(f_rt)
        axhandles.append(ax_rt)

    return np.array(morphVals), pport2, (lRT, rRT), (lRT_mu,rRT_mu),omissions,(fhandles,axhandles)




def plot_learning_curve(R,order = [],title='',toPlot = True):
    '''plot percent correct for each context.
    assuming for now that its only morphs 0 and 1'''

    # sort sessions
    if order == []:
        #sessList = sorted([int(i.split('_')[0]) for i in R.keys()])
        #order = [str(i)+'_LR' for i in sessList]
        order = R.keys()
    # get p(lick port2| morph) for each session
    MV,PR= [],[]
    for s in order:
        mv,pport2, *trash = psychometrics(R[s],plots='')
        MV.append(mv)
        PR.append(pport2)


    # for each morph
    PR = np.array(PR)
    if toPlot:
        f,ax = plt.subplots()
        ax.plot(PR[:,0],'k',PR[:,1],'r')
        plt.legend(['context 1','context 2'])
        ax.set_ylabel('P(lick port2)')
        ax.set_xlabel('session')
        ax.set_title(title)

        return PR, f, ax
    else:
        return PR

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

    def __init__(self,mouse,sessions,basedir="home",exp = "2AFC_V3",scene = "OneSidedCues"):
        if basedir == "home":
            basestr = "/Volumes/mplitt/VR/" + exp + "/" + mouse + "/"
            #basestr = "/Users/mplitt/Dropbox/tmpVRDat/" +mouse + "/"
        elif basedir == "work":
            basestr = "Z:/VR/" + exp + "/" + mouse + "\\"
        elif basedir == "rig":
            basestr = "Z://VR/" + exp + "/" + mouse + "/"
        else:
            raise Exception("Invalid basedir!..options are 'home' or 'work' ")

        self.basestr = basestr



        if len(list(sessions)) == 0: # if no sessions are specified, find all of them

            data_files = glb(basestr + scene + "*_Licks.txt" )


            sessions = [(i.split(basestr)[1]).split("_Licks.txt")[0] for i in data_files]

            overwritten_files = glb(basestr + scene + "*_Licks_copy*.txt")
            if len(overwritten_files) >0:
                raise Exception("Files appear to have been double saved ")
        else:
            for s in list(sessions):
                try:
                    open(basestr+s+"_Licks_copy*.txt")
                    print("WARNING!!!! multiple copies of session %s" % s)
                except:
                    pass




        self.sessions = sessions
        self.mouse = mouse
        self.data = {}


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
        if lickDat.shape[1] == 4:
            twoPort=True
        else:
            twoPort=False


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
        gridData['port1 licks'] = np.zeros([numCaFrames,])
        gridData['port1 licks'][:] = np.nan
        if twoPort:
            gridData['port2 licks'] = np.zeros([numCaFrames,])
            gridData['port2 licks'][:] = np.nan
        #gridData['punishment'] = np.zeros([tGrid.size-1,])
        gridData['morph'] = np.zeros([numCaFrames,])
        #gridData['morph'][:] = np.nan
        gridData['side'] = np.zeros([numCaFrames,])
        #gridData['side'][:] = np.nan
        gridData['rewards'] = np.zeros([numCaFrames,])
        #gridData['rewards'][:] = np.nan
        #gridData['manual rewards'] = np.zeros([numCaFrames,])
        #gridData['manual rewards'][:] = np.nan
        #gridData['time'] = np.arange(0,numCaFrames/fr,fr)
        #print(gridData['time'].shape)
        #print(gridData['position'].shape)
        #gridData['trial'] = np.zeros([tGrid.size-1,])
        gridData['ca_inds'] = np.arange(caInds[0],caInds[-1]+1,1)

        timeSinceStartup = np.zeros([numCaFrames,])
        timeSinceStartup[:] =np.nan
        nonNan = []

        # find tstart_inds before resampling to prevent errors
        tstart_inds_vec = np.zeros([numCaFrames,])
        tstart_inds_raw = np.where(np.ediff1d(posDat[:,0],to_begin = -900)<=-300)[0]
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

            gridData['port1 licks'][final_ind] = lickDat[inds_to_avg,0].sum()
            if twoPort:
                gridData['port2 licks'][final_ind] = lickDat[inds_to_avg,1].sum()
                gridData['rewards'][final_ind] =lickDat[inds_to_avg,2].max()
                timeSinceStartup[final_ind] = lickDat[inds_to_avg,3].mean()
            else:
                gridData['rewards'][final_ind] =lickDat[inds_to_avg,1].max()
                timeSinceStartup[final_ind] = lickDat[inds_to_avg,2].mean()




            tstart_inds_vec[final_ind] = tstart_inds_vec_raw[inds_to_avg].max()
            #if tstart_inds_raw[tstart_counter] in inds_to_avg:
            #    tstart_inds.append(final_ind)
            #    tstart_counter+=1
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

        reward_inds,first_lick_inds = [],[]
        for trial in range(rewardDat.shape[0]):
        #print(rewardDat[trial,:])

            # make morph vector same length as everything else
            if twoPort:
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
            first_lick_inds.append(rval_ind)
            #print(trial,rval)
            #reward_rec_ind = int(np.argmin(np.abs(timeSinceStartup-rewardTime)))

            #rval = np.max(gridData.iloc[tstart_inds[trial]:tstart_inds[trial+1],gridData.columns.get_loc('rewards')].values)
            gridData.iloc[tstart_inds[trial]:tstart_inds[trial]+rval_ind,gridData.columns.get_loc('rewards')] = -rval
            gridData.iloc[tstart_inds[trial]+rval_ind:tstart_inds[trial+1],gridData.columns.get_loc('rewards')] = rval


        #gridData.loc[gridData['position']<0, 'morph'] = -1



        if save_data:
            pass

        #if tstart_inds.shape>len(reward_inds):
        #    return gridData, tstart_inds[:-1], reward_inds, first_lick_inds
        #else:
        return gridData, tstart_inds[:-1], reward_inds, first_lick_inds



    def _interpolate_data(self,sess):
        '''interpolate all behavioral timeseries to 30 Hz common grid'''

        # lick file

        lickDat = np.genfromtxt(self.basestr + sess + "_Licks.txt",dtype='float',delimiter='\t')
        # c_1  c_2 r realtimeSinceStartup



        # reward file
        rewardDat = np.genfromtxt(self.basestr + sess + "_Rewards.txt",dtype='float',delimiter='\t')
        #print(rewardDat.shape)
        #position.z realtimeSinceStartup paramsScript.morph side

        # manual rewards - looks like this may have not saved correctly in LR sessions
        mRewardDat = np.genfromtxt(self.basestr + sess + "ManRewards.txt",dtype='float',delimiter='\t')
        #realtimeSinceStartup side

        posDat = np.genfromtxt(self.basestr + sess + "_Pos.txt",dtype = 'float', delimiter='\t')
        speed = self._calc_speed(posDat[:,0],posDat[:,1])
        # pos.z realtimeSinceStartup


        # lick data will have earliest timepoint
        #  use time since startup to make common grid for the data
        dt = 1./30.
        endT = lickDat[-1,3] + dt
        tGrid = np.arange(0,endT,dt)

        #gridData = np.zeros([tGrid.size,10])
        gridData = {}
        gridData['position'] = np.zeros([tGrid.size-1,])
        gridData['speed'] = np.zeros([tGrid.size-1,])
        gridData['port1 licks'] = np.zeros([tGrid.size-1,])
        gridData['port2 licks'] = np.zeros([tGrid.size-1,])
        #gridData['punishment'] = np.zeros([tGrid.size-1,])
        gridData['morph'] = np.zeros([tGrid.size-1,])
        gridData['rewards'] = np.zeros([tGrid.size-1,])
        gridData['manual rewards'] = np.zeros([tGrid.size-1,])
        gridData['time'] = tGrid[:-1]
        gridData['trial'] = np.zeros([tGrid.size-1,])

        nonNan = []
        for i in range(tGrid.size-1): # find indices that are within time window
            tWin = tGrid[[i, i+1]]

            lickInd = np.where((lickDat[:,3]>=tWin[0]) & (lickDat[:,3] <= tWin[1]))[0]

            rewardInd = np.where((rewardDat[:,1]>=tWin[0]) & (rewardDat[:,1] <= tWin[1]))[0]

            posInd = np.where((posDat[:,1]>=tWin[0])  & (posDat[:,1]<= tWin[1]))[0]
            try:
                mRewardInd = np.where((mRewardDat[:,0]>=tWin[0]) & (mRewardDat[:,0] <= tWin[1]))[0]
            except:
                mRewardInd = np.array([])
            ## build indices of lickDat

            if posInd.size>0:
                # 1) position
                gridData['position'][i] = posDat[posInd,0].mean()

                # 2) speed, calc outside of loop
                gridData['speed'][i] = speed[posInd].mean()

                nonNan.append(i)
            else:

                # 1) position
                #gridData[i,0] = gridData[i-1,0]
                gridData['position'][i] = np.nan

                # 2) speed
                gridData['speed'][i] = np.nan

            if lickInd.size>0:
                # 3) quinine licks
                gridData['port1 licks'][i] = lickDat[lickInd,0].sum()

                # 4) port2 licks
                gridData['port2 licks'][i] = lickDat[lickInd,1].sum()

                # 7) punishment
                #gridData[i,6] = punishDat[lickInd].sum()
                #gridData['punishment'][i] = punishDat[lickInd].sum()





            # 5) reward cam flag
            if rewardInd.size>0:
                #gridData[i,4]  = rewardDat[rewardInd,3].max()
                gridData['rewards'][i] = rewardDat[rewardInd,3].max()
                gridData['morph'][i] = rewardDat[rewardInd,2].max()

            # 6) manual reward flag
            if mRewardInd.size > 0:
                #gridData[i,5] = mRewardDat[mRewardInd,3].min()
                gridData['manual rewards'][i] = mRewardDat[mRewardInd,1].min()

        # 9) time
        #gridData[:,8] = tGrid

        # interp missing position and speed info
        inds = [i for i in range(tGrid.size-1)]
        gridData['position'] = np.interp(inds,[inds[i] for i in nonNan],[gridData['position'][i] for i in nonNan]).tolist()
        gridData['speed'] = np.interp(inds,[inds[i] for i in nonNan],[gridData['speed'][i] for i in nonNan]).tolist()




        # 10) trial number
        try:
            np.genfromtxt('filename 4 trial start file')
            # find tgrid indices of new trials
        except:
            # find teleports and append a trial start
            trialStart = np.where(np.ediff1d(gridData['position'],to_begin = -900, to_end = -900)<=-100)[0]

        for j in range(trialStart.size-1):
            gridData['trial'][trialStart[j]:trialStart[j+1]] = j


        return gridData


    def _find_single_trials(self,sess,gridData):
        '''make dictionary for each variable that contains list of np arrays for each trial'''
        try:
            np.genfromtxt('filename 4 trial start file')
            # find tgrid indices of new trials
        except:
            # find teleports and append a trial start
            trialStart = np.where(np.ediff1d(gridData['position'],to_begin = -900, to_end = -900)<=-300)[0]


        #trialLists = [[] for i in range(trialStart.size)]
        trialLists = {}
        trialLists['position'] = []
        trialLists['speed'] = []
        trialLists['port1 licks'] = []
        trialLists['port2 licks'] = [ ]
        trialLists['rewards'] = [ ]
        trialLists['manual rewards'] = []
        trialLists['punishment'] = []
        trialLists['time'] = []
        trialLists['morph'] = []

        for i in range(trialStart.size-1):
            trialLists['position'].append(gridData['position'][trialStart[i]:trialStart[i+1]])
            trialLists['speed'].append(gridData['speed'][trialStart[i]:trialStart[i+1]])
            trialLists['port1 licks'].append(gridData['port1 licks'][trialStart[i]:trialStart[i+1]])
            trialLists['port2 licks'].append(gridData['port2 licks'][trialStart[i]:trialStart[i+1]])
            trialLists['rewards'].append(gridData['rewards'][trialStart[i]:trialStart[i+1]])
            trialLists['morph'].append(gridData['morph'][trialStart[i]:trialStart[i+1]].max()*np.ones(gridData['rewards'][trialStart[i]:trialStart[i+1]].shape))
            trialLists['manual rewards'].append(gridData['manual rewards'][trialStart[i]:trialStart[i+1]])
            trialLists['punishment'].append(gridData['punishment'][trialStart[i]:trialStart[i+1]])
            trialLists['time'].append(gridData['time'][trialStart[i]:trialStart[i+1]])

        return trialLists



    def _reward_trig_dat(self,dataDict,dt = 1./30.):
        # want reward triggered licks

        #find indices of rewards and take .5 sec prior and 3 secs after
        rewardCamInds = np.where(dataDict['rewards']>0)[0]
        back, forward = int(np.floor(.5/dt)), int(np.floor(3./dt))


        port1Licks, port2Licks = np.zeros([rewardCamInds.size,back+forward]), np.zeros([rewardCamInds.size,back+forward])
        trial, morph, side = np.zeros([rewardCamInds.size,]),  np.zeros([rewardCamInds.size,]),  np.zeros([rewardCamInds.size,])
        for r in range(rewardCamInds.size):
            if rewardCamInds[r]+forward>dataDict['port1 licks'].size:
                excess = rewardCamInds[r]+forward-dataDict['port1 licks'].size


                port1Licks[r,:-excess] = dataDict['port1 licks'][rewardCamInds[r]-back:]
                port2Licks[r,:-excess] = dataDict['port2 licks'][rewardCamInds[r]-back:]
            else:
                port1Licks[r,:] = dataDict['port1 licks'][rewardCamInds[r]-back:rewardCamInds[r]+forward]
                port2Licks[r,:] = dataDict['port2 licks'][rewardCamInds[r]-back:rewardCamInds[r]+forward]



            trial[r] = dataDict['trial'][rewardCamInds[r]]
            morph[r] = dataDict['morph'][rewardCamInds[r]]
            side[r] = dataDict['rewards'][rewardCamInds[r]]

        rewardTrigDat = {}
        rewardTrigDat['port1 licks'] = port1Licks
        rewardTrigDat['port2 licks'] = port2Licks
        rewardTrigDat['trial'] = trial
        rewardTrigDat['morph'] = morph
        rewardTrigDat['side'] = side
        rewardTrigDat['time'] = np.arange(-back*dt,forward*dt,dt)

        rewardTrigDat['first lick'], rewardTrigDat['RT'] = self._response_dat(port1Licks[:,back-1:],port2Licks[:,back-1:])


        return rewardTrigDat





    def _response_dat(self,port1Licks,port2Licks,dt=1./30.):

        firstLicks = np.zeros([port1Licks.shape[0],])
        rt = np.zeros([port1Licks.shape[0],])
        # get first lick direction
        for i in range(port1Licks.shape[0]):

            l1 = np.where(port1Licks[i,:]>0)[0]

            r1 = np.where(port2Licks[i,:]>0)[0]

            if len(l1) == 0 and len(r1) == 0:
                firstLicks[i] = 0
                rt[i] = np.nan
            elif len(l1)>0 and len(r1) == 0:
                firstLicks[i] = 1
                rt[i] = dt*l1[0]
            elif len(l1)==0 and len(r1)>0:
                firstLicks[i] = 2
                rt[i]=dt*r1[0]
            else:
                if l1[0]<r1[0]:
                    firstLicks[i]=1
                    rt[i] = dt*l1[0]
                elif r1[0]<l1[0]:
                    firstLicks[i]=2
                    rt[i]=dt*r1[0]
                else:
                    firstLicks[i] = 0
                    rt[i] = np.nan


        return firstLicks, rt


    def _calc_speed(self,pos,t, toSmooth = True ):
        '''calculate speed from position and time vectors'''
        dt = np.ediff1d(t,to_end=1)
        dt[dt==0.] = np.nan
        rawSpeed = np.divide(np.ediff1d(pos,to_end=0),dt)
        #nanInds = np.where(np.isnan(rawSpeed))
        #for i in nanInds.size:
        #    if i == 0:


        notTeleports = np.where(np.ediff1d(pos,to_begin=0)>-50)[0]

        #rawSpeed[teleports] = np.nan
        inds = [i for i in range(rawSpeed.size)]

        rawSpeed = np.interp(inds,[inds[i] for i in notTeleports],[rawSpeed[i] for i in notTeleports])

        if toSmooth:
            speed = self._smooth_speed(rawSpeed)
        else:
            speed = rawSpeed

        return speed


    def _smooth_speed(self,speed,smooth=10):
        return gaussian_filter(speed,smooth)
