import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import numpy as np
import matplotlib.pyplot as plt
import utilities as u
import preprocessing as pp
import behavior as b

#### fix for new code

def LickPlots(data,save = False, dir = None):
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(data)
    lick_trial_mat,tmp, edges, centers = u.make_pos_bin_trial_matrices(data['lick']._values,
                                    data['pos']._values,
                                    data['tstart']._values,
                                    data['teleport']._values,bin_size=5)
    lick_morph_dict = u.trial_type_dict(lick_trial_mat,trial_info['morphs'])
    max_pos = np.copy(trial_info['max_pos'])
    max_pos[max_pos>440]=np.nan

    f, (ax, meanlr_ax, lickrat_ax) = b.lick_plot(lick_morph_dict,edges,max_pos=max_pos,smooth=False,
                                    rzone1=(250.,315),rzone0=(350,415))
    if save:
        f.savefig(dir + '_lick_plot.pdf',format='pdf')
        f.savefig(dir + '_lick_plot.svg',format='svg')




if __name__ == '__main__':
    # mice = ['4139219.3','4139224.2','4139224.3','4139224.5']
    mice = ['4139261.2','4139251.1','4139260.1']


    df = pp.load_session_db()
    df = df[df['RewardCount']>20]
    df = df[df['Imaging']==1]
    mask = df['ImagingRegion']=="CA1"
    df = df[mask]
    df = df.sort_values(['MouseName','DateTime','SessionNumber'])
    tracks = 'TwoTower_noTimeout|TwoTower_Timeout|Reversal_noTimeout|Reversal|TwoTower_foraging'
    df = df[df['Track'].str.contains(tracks,regex=True)]



    for mouse in mice:

        dirBase = "G:\\My Drive\\Figures\\TwoTower\\Behavior\\%s\\" % mouse
        try:
            os.makedirs(dirBase)
        except:
            print("noTO directory already made")



        df_mouse = df[df['MouseName'].str.match(mouse)]
        df_noTO = df_mouse[df_mouse['Track'].str.match('Reversal_noTimeout')]
        for i in range(int(np.min([3,df_noTO.shape[0]]))):
            fname = "%s\\%s_%s_%d" % (dirBase,mouse,df_noTO['DateFolder'].iloc[i],df_noTO['SessionNumber'].iloc[i])
            print(fname)
            data = pp.behavior_dataframe(df_noTO['data file'].iloc[i])
            LickPlots(data,save=True,dir=fname)

        data = pp.behavior_dataframe([df_noTO['data file'].iloc[i] for i in range(df_noTO.shape[0])],concat=True)
        fname = "%s\%s_all_" % (dirBase,mouse)
        LickPlots(data,save=True,dir=fname)


        mask = df_mouse['Track']=="Reversal"
        df_TO = df_mouse[mask]
        for j in range(df_TO.shape[0]):
            fname = "%s\\%s_%s_%d_" % (dirBase,mouse,df_TO['DateFolder'].iloc[j],df_TO['SessionNumber'].iloc[j])
            data = pp.behavior_dataframe(df_TO['data file'].iloc[j])
            LickPlots(data,save=True,dir=fname)

        data = pp.behavior_dataframe([df_TO['data file'].iloc[i] for i in range(df_TO.shape[0])],concat=False)
        (f_sess,ax_sess), (f_pcntcorr, ax_pcntcorr), (f_lp, ax_lp)=b.learning_curve_plots(data,reversal=True)

        f_sess.savefig(fname = "%s\%s_sess.pdf" % (dirBase,mouse),format='pdf')
        f_sess.savefig(fname = "%s\%s_sess.svg" % (dirBase,mouse),format='svg')


        f_pcntcorr.savefig(fname = "%s\%s_pcntcorr.pdf" % (dirBase,mouse),format='pdf')
        f_pcntcorr.savefig(fname = "%s\%s_pcntcorr.svg" % (dirBase,mouse),format='svg')

        f_lp.savefig(fname = "%s\%s_lp.pdf" % (dirBase,mouse),format='pdf')
        f_lp.savefig(fname = "%s\%s_lp.svg" % (dirBase,mouse),format='svg')
