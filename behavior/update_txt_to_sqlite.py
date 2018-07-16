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
#print(mice)
sess_connect = sql.connect("Z:\\VR\\TwoTower\\behavior.sqlite")

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

        if (ds == ds_old) and (sess == sess_old):
            num+=1
        else:
            num=1
        ds_old = ds
        sess_old = sess


        db_name = "%s%s%s%s%d%s" % (ds,"\\",  sess , "_" , num , ".sqlite")
        print(lf,ds, db_name)
        try:
            conn = sqlite3.connect(ds + "\" + sess + "_" + string(num) + ".sqlite")

            if sess in ["TwoTower_noTimeout", "TwoTower_Timeout"]:
                #print(lf[:-11])
                c = pd(mouse,lf[:-11])
                gridData, (rewardedTrials, errorTrials, omissionTrials, morphList) = c._interpolate_data()
                # connect to new sqlite database

                # for each entry in grid data
                #for i,pos in enumerate(gridData['position'].tolist()):
            elif sess in ["RunningTraining"]:
                pass
                # load lick file and position files
                # make sure they match in lengths

                # create table

                # write each line to table

            elif sess in ["LEDCue","FlashLED"]:
                pass
                # load lick files

                # create table

                # write each line to table
        except:
            print("connection to sqlite database failed")


        if (("imaging" in lf) or ("Imaging" in lf)) and ("post" not in lf) and ("pre" not in lf):
            #print(lf)
            # find imaging file with closest modification date
            os.chdir("Z:\\2P_data\TwoTower\%s" % mouse)
            #Timeout_dirs, noTimeout_dirs = glob('TT_to*/*.sbx'), glob('TT_train*/*.sbx')
            sbx_files = glob('TT_to*/*.sbx') + glob('TT_train*/*.sbx')
            #if "noTimeout" in lf:
            #    ind = np.argmin(np.abs(mt-np.array([os.path.getmtime(d) for d in noTimeout_dirs])))
            #    print(lf, noTimeout_dirs[ind])

            ind = np.argmin(np.abs(mt-np.array([os.path.getmtime(d) for d in sbx_files])))
            scanfile_stem = sbx_files[ind][:-4]
            #print(lf, scanfile_stem)

            # move sbx, mat, & any h5 or .mat  files
            #print(glob(scanfile_stem + "*"))
            try:
                os.mkdir(ds)
            except:
                pass

            for f in glob(scanfile_stem + "*"):
                pass



        # delete old text files
