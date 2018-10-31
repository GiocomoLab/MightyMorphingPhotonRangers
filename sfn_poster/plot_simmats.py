import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import numpy as np
import matplotlib.pyplot as plt
import utilities as u
import preprocessing as pp
import behavior as b
import SimilarityMatrixAnalysis as sm
import pickle





# similarity matrix
S, U, U_norm, (f,ax_S), (f_U, ax_U) = sm.single_session(df_mouse.iloc[-1])
