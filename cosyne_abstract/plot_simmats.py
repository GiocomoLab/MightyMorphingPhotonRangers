import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import numpy as np
import matplotlib.pyplot as plt
import utilities as u
import preprocessing as pp
import behavior as b
import SimilarityMatrixAnalysis as sm
import PlaceCellAnalysis as pc
import pickle
from multiprocessing import Pool


def single_session(a):
    df_sess,mouse,dirbase,i = a[0], a[1], a[2], a[3]
    fname = "%s\\%s_%s_%d_" % (dirbase,mouse,df_sess['DateFolder'].iloc[i],df_sess['SessionNumber'].iloc[i])
    print(fname)


    VRDat,C,Cd, Spikes, A = pp.load_scan_sess(df_sess.iloc[i])
    # get trial by trial info
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    C_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(C,VRDat['pos']._values,VRDat['tstart']._values,VRDat['teleport']._values)
    C_morph_dict = u.trial_type_dict(C_trial_mat,trial_info['morphs'])
    occ_morph_dict = u.trial_type_dict(occ_trial_mat,trial_info['morphs'])

    # find place cells individually on odd and even trials
    # keep only cells with significant spatial information on both
    # print('pc')
    # masks, FR, SI = pc.place_cells_split_halves(C, VRDat['pos']._values,trial_info, VRDat['tstart']._values, VRDat['teleport']._values)


    # similarity matrix
    print('corr')
    S, U, (f,ax_S), (f_U, ax_U) = sm.single_session(df_sess.iloc[i],C=C,VRDat=VRDat,corr=True,bootstrap=True)
    f.savefig(fname+"S_corr.pdf",format='pdf')
    f.savefig(fname+"S_corr.svg",format='svg')
    f_U.savefig(fname+"U_corr.pdf",format='pdf')
    f_U.savefig(fname+"U_corr.svg",format='svg')

    results = {'S':S, 'U':U}
    with open(fname+"results_corr.pkl",'wb') as f:
        pickle.dump(results,f)

    # # similarity matrix place cells only
    # print('corr pc')
    # S, U, (f,ax_S), (f_U, ax_U) = sm.single_session(df_sess.iloc[i],C=C,VRDat=VRDat,corr=True,bootstrap=True,mask=masks[0]+masks[1])
    # f.savefig(fname+"S_corr_pcOR.pdf",format='pdf')
    # f.savefig(fname+"S_corr_pcOR.svg",format='svg')
    # f_U.savefig(fname+"U_corr_pcOR.pdf",format='pdf')
    # f_U.savefig(fname+"U_corr_pcOR.svg",format='svg')
    #
    # results = {'S':S, 'U':U}
    # with open(fname+"results_corr_pcOR.pkl",'wb') as f:
    #     pickle.dump(results,f)

    # similarity matrix
    print('sim')
    S, U, (f,ax_S), (f_U, ax_U) = sm.single_session(df_sess.iloc[i],C=C,VRDat=VRDat,corr=False,bootstrap=True)
    f.savefig(fname+"S.pdf",format='pdf')
    f.savefig(fname+"S.svg",format='svg')
    #f.close()
    f_U.savefig(fname+"U.pdf",format='pdf')
    f_U.savefig(fname+"U.svg",format='svg')
    #f_U.close()
    results = {'S':S, 'U':U}
    with open(fname+"results.pkl",'wb') as f:
        pickle.dump(results,f)



if __name__ == '__main__':
    mice = ['4139212.2','4139212.4','4139219.2','4139219.3','4139224.2','4139224.3','4139224.5']

    df = pp.load_session_db()
    df = df[df['RewardCount']>20]
    df = df[df['Imaging']==1]
    df = df.sort_values(['MouseName','DateTime','SessionNumber'])



    for mouse in mice:

        dirbase = "G:\\My Drive\\Figures\\TwoTower\\COSYNE2019\\SimMats\\%s\\" % mouse
        try:
            os.makedirs(dirbase)
        except:
            print("directory already made")



        df_mouse = df[df['MouseName'].str.match(mouse)]
        df_sess = df_mouse[df_mouse['Track'].str.match('TwoTower_noTimeout') | df_mouse['Track'].str.match('TwoTower_Timeout')]
        args = [[df_sess,mouse,dirbase,i] for i in range(df_sess.shape[0])]
        for i in range(df_sess.shape[0]):
            try:
                single_session(args[i])
            except:
                pass

        #with Pool(8) as p:
        #    p.map(single_session,args)
