import single_session_plots as ssp
import preprocessing as pp



if __name__ == "__main__":

    # mice = ['4139265.3','4139265.5','4139265.4',
    #  '4222154.1','4222154.2','4222153.1','4222153.2','4222153.3',
    #  '4139219.2', '4139219.3', '4139224.2', '4139224.3', '4139224.5',
    #  '4139251.1','4139251.2','4139260.1','4139261.2','4139266.3','4139260.2']
    #mice = ['4222154.1','4222153.1','4222153.2','4222153.3','4139260.2',
    #'4139219.2', '4139219.3', '4139224.2', '4139224.3', '4139224.5',
    # '4139251.1','4139251.2','4139260.1','4139261.2','4139266.3']

    mice = ['4139265.3','4139265.5','4139265.4','4222175.0'
     '4222154.1','4222153.1','4222153.2','4222174.1', '4222157.3',
     '4139219.2', '4139219.3', '4139224.2', '4139224.3', '4139224.5',
     '4139251.1','4139260.1','4139261.2',
     '4139278.2','4222157.4']

    mice = [ '4139224.3', '4139224.5',
    '4139251.1','4139260.1','4139261.2',
    '4139278.2','4222157.4']

    # mice = ['4139266.3']
    df = pp.load_session_db()
    df = df[df['RewardCount']>20]
    df = df[df['Imaging']==1]
    df = df.sort_values(['MouseName','DateTime','SessionNumber'])

    ops = {'behavior':False,
        'PCA':False,
        'place cells':True,
        'trial simmats':False,
        'trial NMF': False,
        'savefigs':True}
    for mouse in mice:
        df_mouse = df[df['MouseName'].str.match(mouse)]
        print(mouse,df_mouse['scanmat'])
        f= []
        for i in range(df_mouse.shape[0]):
            sess = df_mouse.iloc[i]
            try:
                ssp.single_session_figs(sess,ops=ops)
            except:
                print("Fail",sess)
