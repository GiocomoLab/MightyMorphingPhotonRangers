import numpy as np
import scipy as sp
import os

os.sys.path.append("C:\\Users\\mplitt\\MightyMorphingPhotonRangers")
import utilities as u

class SingleCellModel:
    def __init__(self,ops={}}):
        self._set_ops(ops)
        self._set_ctrl_pts()
        s= self.ops['s']
        self.S = np.array([[-s, 2-s, s-2, s],
                    [2*s, s-3, 3-2*s, -s],
                    [-s, 0, s, 0,],
                    [0 1 0 0]])

    def _set_ops(ops_in):
        ops_out={'key':0,
        'n_ctrl_pts_pos':10,
        'n_ctrl_pts_morph':5,
        'max_pos'=450}
        for k,v in ops_in.items():
            ops_out[k]=v

        self.ops = ops_out

    def _set_ctrl_pts():
        self.pos_ctrl_pts = np.linspace(0,self.ops['max_pos'],num=self.ops['n_ctrl_pts_pos'])
        self.morph_ctrl_pts = np.linspace(0,1,num=self.ops['n_ctrol_pts_morph'])


    def pos_morph_spline(pos,morph):
        assert pos.shape==morph.shape, "position and morph vectors need to be of same length"

        X = np.zeros(self)
        for i in range(pos.shape[0]):
            p,m = pos[i],morph[i]
            x_p = self._1d_spline_coeffs(self.pos_ctrl_pts,p)
            x_m = self._1d_spline_coeffs(self.morph_ctrl_pts,m)

            xx = np.matmul(x_p.reshape([-1,1]),x_m.reshape([1,-1]))


        return X

    def _1d_spline_coeffs(ctrl,v):
        # need to pad original ctrl point vector
        ### move this part to _set_ctrl_pts
        binW = ctrl[1]-ctrl[0]
        cctrl = np.append(np.insert(ctrl,0,ctrl[0]-binW),ctrl[-1]+binW)
        x = np.zeros(cctrl.shape)

        # nearest ctrl pt
        cctrl_i = (cctrl<v).sum()-1
        pre_ctrl_pt = cctrl[cctrl_i]

        # next ctrl pt
        post_ctrl_pt = cctrl[cctrl_i+1]

        alpha = (v-pre_ctrl_pt)/(post_ctrl_pt-pre_ctrl_pt)
        u = np.array([alpha**3, alpha**2, alpha, 1]).reshape([1,-1])
        # p =
        x[cctrl_i-1:cctrl_i+2] = np.matmult(u,self.S)
        return x


# %for each timepoint, calculate the corresponding row of the glm input matrix
# for i=1:length(x1)
#
#     % for 1st dimension
#     % find the nearest, and next, control point
#     nearest_c_pt_index_1 = max(find(cpts_all < x1(i)));
#     nearest_c_pt_time_1 = cpts_all(nearest_c_pt_index_1);
#     next_c_pt_time_1 = cpts_all(nearest_c_pt_index_1+1);
#
#     % compute the alpha (u here)
#     u_1 = (x1(i)-nearest_c_pt_time_1)/(next_c_pt_time_1-nearest_c_pt_time_1);
#     p_1=[u_1^3 u_1^2 u_1 1]*S;
#
#     % fill in the X matrix, with the right # of zeros on either side
#     X1 = [zeros(1,nearest_c_pt_index_1-2) p_1 zeros(1,num_c_pts-4-(nearest_c_pt_index_1-2))];
#
#
#
#     % for 2nd dimension
#     % find the nearest, and next, control point
#     nearest_c_pt_index_2 = max(find(cpts_all < x2(i)));
#     nearest_c_pt_time_2 = cpts_all(nearest_c_pt_index_2);
#     next_c_pt_time_2 = cpts_all(nearest_c_pt_index_2+1);
#
#     % compute the alpha (u here)
#     u_2 = (x2(i)-nearest_c_pt_time_2)/(next_c_pt_time_2-nearest_c_pt_time_2);
#     p_2=[u_2^3 u_2^2 u_2 1]*S;
#
#     % fill in the X matrix, with the right # of zeros on either side
#     X2 = [zeros(1,nearest_c_pt_index_2-2) p_2 zeros(1,num_c_pts-4-(nearest_c_pt_index_2-2))];
#
#     % take the outer product
#     X12_op = X2'*X1; X12_op = flipud(X12_op);
#
#     X(i,:) = X12_op(:);
#
# end
