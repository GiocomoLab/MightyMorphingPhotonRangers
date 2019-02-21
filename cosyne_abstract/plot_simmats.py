import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import numpy as np
import matplotlib.pyplot as plt
import utilities as u
import preprocessing as pp
import SimilarityMatrixAnalysis as sm
import pickle



def single_session(sess,dirbase):

    fname = "%s\\%s_%d_" % (dirbase,sess['DateFolder'],sess['SessionNumber'])
    print(fname)

    # similarity matrix
    print('corr')
    SM, U, (f_S,ax_S), (f_U, ax_U) = sm.single_session(sess)
    f_S.savefig(fname+"S.pdf",format='pdf')
    f_U.savefig(fname+"U.pdf",format='pdf')


    results = {'S':S, 'U':U}
    with open(fname+"results.pkl",'wb') as f:
        pickle.dump(results,f)



if __name__ == '__main__':

    mice = ['4139219.2','4139219.3','4139224.2','4139224.3','4139224.5',
    '4139251.1','4139260.1','4139260.2','4139261.2','4139265.3','4139265.4',
    '4139265.5','4139266.3']


    df = pp.load_session_db()
    df = df[df['RewardCount']>20]
    df = df[df['Imaging']==1]
    df = df.sort_values(['MouseName','DateTime','SessionNumber'])
    tracks = 'TwoTower_noTimeout|TwoTower_Timeout|Reversal_noTimeout|Reversal|TwoTower_foraging'
    df = df[df['Track'].str.contains(tracks,regex=True)]


    for mouse in mice:
        dirbase = "G:\\My Drive\\Figures\\TwoTower\\SimMats\\%s\\" % mouse

        try:
            os.makedirs(dirbase)
        except:
            print("directory already made")

        df_mouse = df[df['MouseName'].str.match(mouse)]
        for i in range(df_mouse.shape[0]):
            sess = df_mouse.iloc[i]
            try:
                single_session(sess,dirbase)
            except:
                print(sess)
