import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import numpy as np
import matplotlib.pyplot as plt
import utilities as u
import preprocessing as pp
import behavior as b
import SimilarityMatrixAnalysis as sm
import pickle



if __name__ == '__main__':
#     mice = ['4139190.1', '4139190.3','4139212.2','4139212.4','4139219.2','4139219.3','4139224.2','4139224.3','4139224.5']
    mice = ['4139224.2','4139224.3','4139224.5']

    df = pp.load_session_db()
    df = df[df['RewardCount']>20]
    df = df[df['Imaging']==1]
    df = df.sort_values(['MouseName','DateTime','SessionNumber'])



    for mouse in mice:

        dirbase = "G:\\My Drive\\Figures\\TwoTower\\SimMats\\%s\\" % mouse
        try:
            os.makedirs(dirbase)
        except:
            print("directory already made")



        df_mouse = df[df['MouseName'].str.match(mouse)]
        df_sess = df_mouse[df_mouse['Track'].str.match('TwoTower_noTimeout') | df_mouse['Track'].str.match('TwoTower_Timeout')]
        for i in range(df_sess.shape[0]):
            try:
                fname = "%s\\%s_%s_%d_" % (dirbase,mouse,df_sess['DateFolder'].iloc[i],df_sess['SessionNumber'].iloc[i])
                print(fname)

                # similarity matrix
                S, U, (f,ax_S), (f_U, ax_U) = sm.single_session(df_sess.iloc[i])
                f.savefig(fname+"S.pdf",format='pdf')
                f.savefig(fname+"S.svg",format='svg')
                f_U.savefig(fname+"U.pdf",format='pdf')
                f_U.savefig(fname+"U.svg",format='svg')

                results = {'S':S, 'U':U}
                with open(fname+"results.pkl",'wb') as f:
                    pickle.dump(results,f)
            except:
                pass
