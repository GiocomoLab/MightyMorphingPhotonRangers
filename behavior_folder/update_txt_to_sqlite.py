import sqlite3 as sql
import sys
import os
import time
import datetime
from glob import glob
import numpy as np
from behavior_two_towers import process_data as pd


# for each mouse
os.chdir('Z:\VR\TwoTower')
mice = [p.replace('\\','') for p in glob('*/')]
#mice = ['4054010.4','4139206.1','4139190.1','4139190.3','4054011.1','4054010.5','4054011.2']
#print(mice)
sess_connect = sql.connect("Z:\\VR\\TwoTower\\behavior.sqlite")
sess_c = sess_connect.cursor()

for mouse in mice:
    print(mouse)
    # find all files with *Licks*.txt
    os.chdir("Z:\VR\TwoTower\%s" % mouse)
    lick_files = glob('*Licks*') + glob('*/*Licks*')
    #print(lick_files)
    # get date modified
    mtimes = [os.path.getmtime(f) for f in lick_files]

    # sort by date modified
    mtimes_argsort = sorted(range(len(mtimes)),key=mtimes.__getitem__)
    lick_files, mtimes = [lick_files[m] for m in mtimes_argsort], [mtimes[m] for m in mtimes_argsort]
    date_strs = [datetime.datetime.fromtimestamp(t).strftime("%d_%m_%Y") for t in mtimes]

    # for each file
    ds_old = ''
    sess_old = ''
    num = 1
    for i,(lf, ds, mt) in enumerate(zip(lick_files, date_strs, mtimes)):
        # make folder with date modified
        #print(lf, ds, mt)
        os.chdir("Z:\VR\TwoTower\%s" % mouse)
        try:
            os.mkdir(ds)
        except:
            pass

        if ("1PortTower_noTimeout" in lf) or ("TwoTower_noTimeout" in lf):
            sess = "TwoTower_noTimeout"
        elif ("1PortTower_Timeout" in lf) or ("TwoTower_Timeout" in lf):
            sess = "TwoTower_Timeout"
        elif ("MovingEndWall" in lf):
            sess = "RunningTraining"
        elif ("FlashLED" in lf):
            sess = "FlashLED"
        elif ("LEDCue" in lf):
            sess = "LEDCue"

        #if (ds == ds_old) and (sess == sess_old):
        #    num+=1
        #else:
        #    num=1



        while (True):
            db_name = "%s\\%s_%d.sqlite" % (ds,sess , num )
            if os.path.exists(db_name):
                num+=1
            else:
                num_old = num
                num=1
                break






        print(lf,ds, db_name)
        lf_parts = lf.partition('_Licks')
        #print(lf_parts)
        #try:
        if False:
            conn = sql.connect(db_name)
            c = conn.cursor()
            if sess in ["TwoTower_noTimeout", "TwoTower_Timeout"]:
                #print(lf_parts)
                b = pd(mouse,lf_parts[0])
                gridData = b.to_sql_dicts()

                c.execute('''CREATE TABLE data (time REAL, morph REAL, trialnum INT, pos REAL, dz REAL, lick INT, reward INT,
                    tstart INT, teleport INT, rzone INT, toutzone INT, clickOn NUMERIC, blockWalls NUMERIC, towerJitter REAL,
                    wallJitter REAL, bckgndJitter REAL, scanning NUMERIC, manrewards INT)''')
                # for each entry in grid data
                nRewards = np.sum(gridData['rewards'])
                for i,pos in enumerate(gridData['position'].tolist()):

                    insertStr = '''INSERT INTO data (time , morph, pos, dz, lick, reward,tstart,
                    teleport, clickOn, towerJitter, wallJitter , bckgndJitter , manrewards)
                    VALUES (%f,%f,%f,%f,%d,%d,%d,%d,%d,%f,%f,%f,%d)''' % (gridData['time'][i],gridData['morph'][i],
                                 gridData['position'][i], gridData['delta z'][i],
                                 gridData['licks'][i], gridData['rewards'][i],
                                 gridData['tstart'][i],gridData['teleports'][i],
                                 gridData['click on'][i],gridData['tower jitter'][i],
                                 gridData['wall jitter'][i],gridData['background jitter'][i],
                                 gridData['man rewards'][i])
                    c.execute(insertStr)
            elif sess in ["RunningTraining"]:

                # load lick file and position files
                # make sure they match in lengths
                lickDat = np.genfromtxt(lf,delimiter='\t')

                posDat = np.genfromtxt(lf_parts[0]+"_Pos"+lf_parts[2],delimiter='\t')


                try:
                    manRewardDat = np.reshape(np.genfromtxt(lf_parts[0] + "ManRewards.txt", delimiter='\t'),[-1,2])
                except:
                    manRewardDat = np.array([])

                # find teleport and tstart_inds before resampling to prevent errors
                tstart_inds_vec,teleport_inds_vec = np.zeros([posDat.shape[0],]), np.zeros([posDat.shape[0],])
                teleport_inds = np.where(np.ediff1d(posDat[:,0])<=-50)[0]
                tstart_inds = np.append([0],teleport_inds[:-1]+1)
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

                trialNumber = np.cumsum(tstart_inds_vec)

                mRewards = np.zeros([posDat.shape[0],])
                #print(manRewardDat.shape)
                if manRewardDat.shape[0]>0:
                    for row in range(manRewardDat.shape[0]):
                        mInd = np.argmin(np.abs(posDat[:,1]-manRewardDat[row,0]))
                        mRewards[mInd] = 1

                c.execute('''CREATE TABLE data (time REAL, trialnum INT, pos REAL, dz REAL, lick INT, reward INT,
                    tstart INT, teleport INT, scanning NUMERIC, manrewards INT)''')

                nRewards=np.sum(lickDat[:,0])
                for i in range(np.min([lickDat.shape[0],posDat.shape[0]])):
                    insertstr = '''INSERT INTO data (time , trialnum, pos, dz, lick, reward,
                        tstart, teleport,  manrewards) VALUES (%g, %d, %g ,%g, %d, %d,
                        %d, %d, %d)''' % (posDat[i,1], trialNumber[i], posDat[i,0], posDat[i,-1],
                        lickDat[i,0], lickDat[i,1], tstart_inds_vec[i], teleport_inds_vec[i],mRewards[i])
                    c.execute(insertstr)

            elif sess in ["FlashLED"]:
                # load lick files
                lickDat = np.genfromtxt(lf,delimiter='\t')
                nRewards = np.sum(lickDat[:,0])
                # create table
                c.execute('''CREATE TABLE data (time REAL, LEDCue INT, dz REAL, lick INT,
                 reward INT, gng INT, scanning NUMERIC, manrewards INT)''')

                # write each line to table
            elif sess in ["LEDCue"]:
                lickDat = np.genfromtxt(lf,delimiter='\t')
                nRewards = np.sum(lickDat[:,0])
                c.execute('''CREATE TABLE data (time REAL, dz REAL, lick INT, reward INT, manrewards INT)''')

            conn.commit()
            conn.close()

        if (("imaging" in lf) or ("Imaging" in lf)) and ("post" not in lf) and ("pre" not in lf):
            imaging = 1
            # find imaging file with closest modification date
            os.chdir("Z:\\2P_data\TwoTower\%s" % mouse)
            #Timeout_dirs, noTimeout_dirs = glob('TT_to*/*.sbx'), glob('TT_train*/*.sbx')
            sbx_files = glob('TT_to*/*.sbx') + glob('TT_train*/*.sbx')
            if len(sbx_files)>0:
                ind = np.argmin(np.abs(mt-np.array([os.path.getmtime(d) for d in sbx_files])))
                scanfile_stem = sbx_files[ind][:-4]
                #print(lf, scanfile_stem)

                # move sbx, mat, & any h5 or .mat  files


                #print(glob(scanfile_stem + "*"))
                try:
                    os.makedirs("%s/%s" % (ds,sess))
                except:
                    pass

                try:

                    os.rename("%s.mat"  % scanfile_stem,"%s/%s/%s_%d_000.mat" % (ds,sess,sess,num_old ))
                    os.rename("%s.sbx"  % scanfile_stem,"%s/%s/%s_%d_000.sbx" % (ds,sess,sess,num_old ))
                except:
                    print("sbx renaming failed")
        else:
            imaging=0


        #sess_c.execute("INSERT INTO sessions (MouseName, DateFolder, SessionNumber, Track, RewardCount, Imaging) VALUES ('%s','%s',%d,'%s',%d,%d)" % (mouse,ds,num_old,sess,nRewards,imaging))

#sess_connect.commit()
#sess_connect.close()
        # delete old text files
