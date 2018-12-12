import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import numpy as np
import matplotlib.pyplot as plt
import utilities as u
import preprocessing as pp
import behavior as b
import PlaceCellAnalysis as pc
import pickle


def getfrac():

    mice = ['4139190.1', '4139190.3','4139212.2','4139212.4','4139219.2','4139219.3','4139224.2','4139224.3','4139224.5']
    #mice = ['4139212.4','4139219.2','4139219.3','4139224.2','4139224.3','4139224.5']
    frac={}
    df = pp.load_session_db()
    df = df[df['RewardCount']>20]
    df = df[df['Imaging']==1]
    df = df.sort_values(['MouseName','DateTime','SessionNumber'])


    frac={}
    for mouse in mice:
            frac[mouse]={}
            dirbase = "G:\\My Drive\\Figures\\TwoTower\\COSYNE2019\\PlaceCells\\Cthr\\%s\\" % mouse

            df_mouse = df[df['MouseName'].str.match(mouse)]
            df_sess = df_mouse[df_mouse['Track'].str.match('TwoTower_noTimeout') | df_mouse['Track'].str.match('TwoTower_Timeout')]
            df_sess = df_sess[df_sess['ImagingRegion'].str.match('CA1') | df_sess['ImagingRegion'].str.match('')]

            frac[mouse][0]=[]
            frac[mouse][1]=[]
            frac[mouse]['common']=[]
            for i in range(df_sess.shape[0]):
                try:
                    fname = "%s\\%s_%s_%d_" % (dirbase,mouse,df_sess['DateFolder'].iloc[i],df_sess['SessionNumber'].iloc[i])
                    print(fname)
                    with open(fname+"results.pkl",'rb') as f:
                        res=pickle.load(f)
                    masks = res['masks']

                    if masks[1].sum()>0:
                        frac[mouse][0].append(masks[0].sum()/masks[0].shape[0])
                        frac[mouse][1].append(masks[1].sum()/masks[1].shape[0])

                    # number with place fields in both
                    common_pc = np.multiply(masks[0],masks[1])
                    frac[mouse]['common'].append(common_pc.sum()/masks[1].shape[0])

                except:
                    print(df_sess.iloc[i])

    return frac
