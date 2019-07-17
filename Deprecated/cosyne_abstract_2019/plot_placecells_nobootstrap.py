import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import numpy as np
import matplotlib.pyplot as plt
import utilities as u
import preprocessing as pp
import behavior as b
import PlaceCellAnalysis as pc
import pickle


if __name__ == '__main__':

    mice = ['4139219.2','4139219.3','4139224.2','4139224.3','4139224.5',
    '4139251.1','4139260.1','4139260.2','4139261.2','4139265.3','4139265.4',
    '4139265.5']

    df = pp.load_session_db()
    df = df[df['RewardCount']>20]
    df = df[df['Imaging']==1]
    df = df.sort_values(['MouseName','DateTime','SessionNumber'])



    for mouse in mice:

            dirbase = "G:\\My Drive\\Figures\\TwoTower\\PlaceCells\\S_nobs\\%s\\" % mouse
            try:
                os.makedirs(dirbase)
            except:
                print("directory already made")



            df_mouse = df[df['MouseName'].str.match(mouse)]
            # df_sess = df_mouse[df_mouse['Track'].str.match('TwoTower_noTimeout') | df_mouse['Track'].str.match('TwoTower_Timeout')]
            for i in range(df_mouse.shape[0]):
                sess = df_mouse.iloc[i]
                try:
                    fname = "%s\\%s_%s_%d_" % (dirbase,mouse,sess['DateFolder'],sess['SessionNumber'])
                    print(fname)

                    FR, masks, SI = pc.single_session(sess,savefigs=True,
                                        deconv=True,fbase =fname,method='SI')
                    results = {'FR':FR, 'masks':masks,'SI':SI}
                    with open(fname+"results.pkl",'wb') as f:
                        pickle.dump(results,f)
                except:
                    print(sess)
