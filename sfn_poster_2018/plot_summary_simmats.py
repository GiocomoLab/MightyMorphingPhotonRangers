import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import numpy as np
import matplotlib.pyplot as plt
import utilities as u
import preprocessing as pp
import behavior as b
import SimilarityMatrixAnalysis as sm
import pickle


# plot no timeout session vs timeout session avg
def getsummarydict():
    mice = ['4139190.1', '4139190.3','4139212.2','4139212.4','4139219.2','4139219.3','4139224.2','4139224.3','4139224.5']

    summ = {}
    df = pp.load_session_db()
    df = df[df['RewardCount']>20]
    df = df[df['Imaging']==1]
    df = df.sort_values(['MouseName','DateTime','SessionNumber'])



    for mouse in mice:

        smdirbase = "G:\\My Drive\\Figures\\TwoTower\\SFN2018\\SimMats\\%s\\" % mouse
        pcdirbase = "G:\\My Drive\\Figures\\TwoTower\\SFN2018\\PlaceCells\\%s\\" % mouse

        df_mouse = df[df['MouseName'].str.match(mouse)]
        df_sess = df_mouse[df_mouse['Track'].str.match('TwoTower_Timeout')]
        df_sess = df_sess[df_sess['ImagingRegion'].str.match('CA1') | df_sess['ImagingRegion'].str.match('')]

        summ[mouse]={}
        summ[mouse]['N'] = []
        summ[mouse]['S'] = []
        summ[mouse]['U']=[]
        summ[mouse]['U_norm']=[]
        for i in range(df_sess.shape[0]):
            try:
                pcfname = "%s\\%s_%s_%d_" % (pcdirbase,mouse,df_sess['DateFolder'].iloc[i],df_sess['SessionNumber'].iloc[i])
                smfname = "%s\\%s_%s_%d_" % (smdirbase,mouse,df_sess['DateFolder'].iloc[i],df_sess['SessionNumber'].iloc[i])

                with open(pcfname+"results.pkl",'rb') as f:
                    pc_res= pickle.load(f)

                summ[mouse]['N'].append(pc_res['masks'][0].shape[0])

                with open(smfname)+"results.pkl",'rb') as f:
                    sm_res = pickle.load(f)
                summ[mouse]['S'].append(sm_res['S'])
                summ[mouse]['U'].append(sm_res['U'])
                summ[mouse]['U_norm'].append(sm_res['U_norm'])

            except:
                pass

    return summ
