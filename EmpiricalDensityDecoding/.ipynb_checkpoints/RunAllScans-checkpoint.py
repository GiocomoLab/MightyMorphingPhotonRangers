import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import numpy as np
import matplotlib.pyplot as plt
import utilities as u
import preprocessing as pp
import behavior as b
import BayesianDecoding as bd
import shutil



if __name__ == '__main__':
    mice = ['4139212.2','4139219.2','4139219.3','4139224.2','4139224.3','4139224.5']
    #mice = ['4139212.4','4139219.2','4139219.3','4139224.2','4139224.3','4139224.5']

    df = pp.load_session_db()
    df = df[df['RewardCount']>20]
    df = df[df['Imaging']==1]
    df = df.sort_values(['MouseName','DateTime','SessionNumber'])



    for mouse in mice:


        df_mouse = df[df['MouseName'].str.match(mouse)]
        df_sess = df_mouse[df_mouse['Track'].str.match('TwoTower_noTimeout') | df_mouse['Track'].str.match('TwoTower_Timeout')]
        for i in range(df_sess.shape[0]):
            #try:
                sess = df_sess.iloc[i]
                prefix = os.path.join("E:\\", "%s_%s_%d" % (sess.MouseName,sess.DateFolder,sess.SessionNumber))
                gdrive = os.path.join("G:\\My Drive\\NonParametricDecoding\\data", "%s_%s_%d" % (sess.MouseName,sess.DateFolder,sess.SessionNumber))
                ss = bd.single_session(sess,prefix=prefix)
                ss.prefix = "E:\\"
                L = ss.likelihood_maps()
                ss.prefix = prefix
                LLR, LLR_pop = ss.ctxt_LLR(L=L,save=True)
                decode_dict = ss.run_decoding(L=L)
                del L
                del decode_dict
                os.system("move %s/ctxt_LLR.npz '%s/ctxt_LLR.npz'" % (prefix,gdrive))
                os.system("move %s/pop_post_i.dat '%s/pop_post_i.dat'" % (prefix,gdrive))
                os.system("move %s/pop_post_ix.dat '%s/pop_post_ix.dat'" % (prefix,gdrive))
                os.system("move %s/post_i.dat '%s/post_i.dat'" % (prefix,gdrive))
                os.system("del %s" % (prefix))
                os.system("del E:\\L.dat")

                #shutil.move(prefix,gdrive)

            #except:
            #    print(sess)
