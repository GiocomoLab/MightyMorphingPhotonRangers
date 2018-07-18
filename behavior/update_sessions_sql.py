import sqlite3 as sql
import sys
import os
import time
import datetime
from glob import glob
import numpy as np
from behavior_two_towers import process_data as pd
import re
import pandas as pd


# for each mouse
os.chdir('Z:\VR\TwoTower')
mice = [p.replace('\\','') for p in glob('*/')]
#print(mice)
sess_connect = sql.connect("Z:\\VR\\TwoTower\\behavior.sqlite")
sess_c = sess_connect.cursor()

for mouse in mice:
    #print(mouse)
    # find all files with *Licks*.txt
    os.chdir("Z:\\VR\TwoTower\%s" % mouse)
    date_folders = [p.replace('\\','') for p in glob('*/')]

    for ds in date_folders:
        #print(ds)
        os.chdir("Z:\\VR\TwoTower\%s\%s" % (mouse,ds))
        db_files = glob('*.sqlite')
        for s in db_files:
            # find integer in string
            num_str = re.search(r'\d+',s).group()
            # partition string using the integer
            parts = s.partition("_%s" % num_str)
            print(os.getcwd())
            print(mouse,ds,s,parts)
            # sess is firstI
            sess = parts[0]
            # num is second
            num = int(num_str)
            # load sql database as pandas dataframe
            conn = sql.connect("Z:\\VR\TwoTower\%s\%s\%s" % (mouse,ds,s))
            curs = conn.cursor()
            curs.execute("select name from sqlite_master where type='table';")
            print(curs.fetchall())
            df = pd.read_sql("SELECT reward FROM data",conn)
            if parts[0][:-1] not in ['FlashLED','LEDCue']:
                numRewards = np.sum(df['reward'].values)
            else:
                numRewards=0
            # sum rewards column

    # check if imaging file exists


            print("Z:\\2P_data\TwoTower\%s\%s\%s\%s_*%s_*.mat" % (mouse,ds,parts[0],parts[0],parts[1][1:]))
            matFiles = glob("Z:\\2P_data\TwoTower\%s\%s\%s\%s_*%s_*.mat" % (mouse,ds,parts[0],parts[0],parts[1][1:]))

            if len(matFiles)>0:
                imaging = 1
            else:
                imaging = 0

            print("imaging=%d" % imaging)

            sess_c.execute("INSERT INTO sessions (MouseName, DateFolder, SessionNumber, Track, RewardCount, Imaging) VALUES ('%s','%s',%d,'%s',%d,%d)" % (mouse,ds,num,sess,numRewards,imaging))

sess_connect.commit()
sess_connect.close()
        # delete old text files
