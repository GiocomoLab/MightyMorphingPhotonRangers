import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
import pickle


os.sys.path.append("C:\\Users\\mplitt\\MightyMorphingPhotonRangers")
import PlaceCellAnalysis as pc
import utilities as u
import preprocessing as pp
import behavior as b
import BayesianDecoding as bd


df = pp.load_session_db()


#df = df[df['RewardCount']>30]
df = df[df['Imaging']==1]
df = df.sort_values(['MouseName','DateTime','SessionNumber'])
df = df[df['RewardCount']>30]


mice = ['4139190.1','4139190.3','4139212.4']

for mouse in mice:
    df_mouse = df[df['MouseName'].str.match(mouse)]
    df_mouse = df_mouse[df_mouse['Track'].str.match('TwoTower_noTimeout') | df_mouse['Track'].str.match('TwoTower_Timeout')]

    for i in range(df_mouse.shape[0]):
        sess = df_mouse.iloc[i]
        prefix = os.path.join("G:\\My Drive\\Figures\\TwoTower\\NonParametricDecoding","%s_%s_%d" %(sess.MouseName, sess.DateFolder, sess.SessionNumber))

        s = bd.single_session(sess,prefix=prefix)

        path = os.path.join("G:\\My Drive\\NonParametricDecoding\\data","%s_%s_%d" %(sess.MouseName, sess.DateFolder, sess.SessionNumber))
        ctxt_LLR = np.load(os.path.join(path,"ctxt_LLR.npz"))

        LLR = ctxt_LLR['arr_0']
        LLR_pop=ctxt_LLR['arr_1']
        pop_post_i = np.fromfile(os.path.join(path,"pop_pos_i.dat"),dtype='float32')
        pop_post_ix = np.reshape(np.fromfile(os.path.join(path,"pop_post_ix.dat"),dtype='float32'),[s.C_z.shape[0],-1])

        post_i = np.reshape(np.fromfile(os.path.join(path,"post_i.dat"),dtype='float32'),[s.C_z.shape[0],-1])

        pc_path = os.path.join("G:\\My Drive\\Figures\\TwoTower\\COSYNE2019\\PlaceCells\\",mouse,
                               "%s_%s_%d_results.pkl" %(mouse,sess.DateFolder,sess.SessionNumber))
        with open(pc_path,'rb') as f:
            pc_res = pickle.load(f)
        cellsort = np.argsort(np.argmax(pc_res['FR'][0]['all']+pc_res['FR'][1]['all'],axis=0))

        s.plot_decoding({'cell i': post_i,'pop ix':pop_post_ix,'pop i':pop_post_i},cellsort=cellsort,save=True)
        cm,fig=s.confusion_matrix({'pop ix':pop_post_ix},save=True)

        np.save(os.path.join(path,"confusion_matrix.npy"),cm)

        mp,mt,pos,time = s.plot_llr(LLR_pop,save=True)


        # plot a random 200 cells
        llrdir = os.path.join(prefix,"cell_llr")
        postdir = os.path.join(prefix,"cell_post_i")
        try:
            os.makedirs(llrdir)
            os.makedirs(postdir)

        except:
            pass

        cell_trial_mat,o,edges,centers = u.make_pos_bin_trial_matrices(post_i,s.pos,s.tstarts,s.teleports)
        cell_d = u.trial_type_dict(cell_trial_mat,s.trial_info['morphs'])
        cell_l_mat,o,edges,centers = u.make_pos_bin_trial_matrices(LLR,s.pos,s.tstarts,s.teleports)
        cell_l_d = u.trial_type_dict(cell_l_mat,s.trial_info['morphs'])

        keys = np.unique(s.trial_info['morphs'])
        perm = np.random.permutation(s.C_z.shape[1])[:200]
        for c in perm.tolist():
            f, ax = plt.subplots(1,keys.shape[0],figsize=[4*keys.shape[0],4])
            ff, aax = plt.subplots(1,keys.shape[0],figsize=[4*keys.shape[0],4])
            for a,m in enumerate(keys.tolist()):
                ax[a].imshow(np.squeeze(cell_d[m][:,:,c]),cmap='cool',vmin=0,vmax=1,aspect='auto')
                aax[a].imshow(np.squeeze(cell_l_d[m][:,:,c]),cmap='cool',vmin=-5,vmax=5,aspect='auto')


            f.savefig(os.path.join(postdir,"cell%d.png" % c),format="png")
            ff.savefig(os.path.join(llrdir,"cell%d.png" % c),format="png")
