# discriminator.py

import torch
import torch.nn as nn
import numpy as np
import os


def normalize_weights(weights):
    num = weights.shape[0]
    channel = weights.shape[1]
    #weights.normal_()
    weights_norm = weights.pow(2).sum(1, keepdim = True).add(1e-8).sqrt()
    weights =  weights/ weights_norm.view(num, 1).repeat(1, channel)

def index_selection_nd(x, I, dim):
    target_shape = [*x.shape]
    del target_shape[dim]
    target_shape[dim:dim] = [*I.shape]
    return x.index_select(dim, I.view(-1)).reshape(target_shape)

class LASMConvssw(nn.Module):
    def __init__(self, in_channel, out_channel, weight_num, in_point_num, connection_info, b_Perpt_bias = True, residual_rate = 0.0): #layer_info_lst= [(point_num, feature_dim)]
        super(LASMConvssw, self).__init__()

        self.relu = nn.ELU()
        # self.norm = nn.GroupNorm(out_channel//4, out_channel)
        # self.norm = nn.BatchNorm1d(out_channel)

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight_num = weight_num
        self.in_point_num = in_point_num
        out_point_num = connection_info.shape[0]
        self.out_point_num = out_point_num

        # if self.in_point_num > self.out_point_num:
        #     self.dropout = nn.Dropout(p=0.5)
        # else:
        #     self.dropout = None

        neighbor_num_lst = torch.from_numpy(connection_info[:,0].astype(np.float32)).float() #out_point_num*1
        self.register_buffer("neighbor_num_lst", neighbor_num_lst)

        neighbor_id_dist_lstlst = connection_info[:, 1:] #out_point_num*(max_neighbor_num*2)
        neighbor_id_lstlst = neighbor_id_dist_lstlst.reshape((out_point_num, -1,2))[:,:,0] #out_point_num*max_neighbor_num

        neighbor_id_lstlst = torch.from_numpy(neighbor_id_lstlst).long()
        self.register_buffer("neighbor_id_lstlst", neighbor_id_lstlst)

        max_neighbor_num = neighbor_id_lstlst.shape[1]
        self.max_neighbor_num = max_neighbor_num

        avg_neighbor_num= round(neighbor_num_lst.mean().item())
        self.avg_neighbor_num = avg_neighbor_num


        ####parameters for conv###############
        weights = nn.Parameter(torch.randn(weight_num, out_channel*in_channel))
        self.register_parameter("weights",weights)

        bias = nn.Parameter(torch.zeros(out_channel))
        if b_Perpt_bias:
            bias= nn.Parameter(torch.zeros(out_point_num, out_channel))

        self.register_parameter("bias",bias)

        self.residual_rate = residual_rate

        ####parameters for residual###############
        #residual_layer = ""
        if self.residual_rate > 0:

            if(out_point_num != in_point_num):
                p_neighbors = nn.Parameter(torch.randn(out_point_num, max_neighbor_num)/(avg_neighbor_num))
                self.register_parameter("p_neighbors",p_neighbors)

            if(out_channel != in_channel):
                weight_res = torch.randn(1, out_channel*in_channel)
                weight_res = weight_res/out_channel

                weight_res = nn.Parameter(weight_res)
                self.register_parameter("weight_res",weight_res)


        # print ("in_channel", in_channel,\
        #     "out_channel",out_channel, \
        #     "in_point_num", in_point_num, \
        #     "out_point_num", out_point_num, \
        #     "weight_num", weight_num,\
        #     "max_neighbor_num", max_neighbor_num)


    # improved version which takes less mem
    def forward(self, in_pc, raw_w_weights, is_final_layer=False, b_max_pool = False):
        batch = in_pc.shape[0]
        device = in_pc.device #in_pc.device

        in_channel = self.in_channel
        out_channel = self.out_channel
        in_pn = self.in_point_num
        out_pn = self.out_point_num
        weight_num = self.weight_num  #M
        max_neighbor_num = self.max_neighbor_num #N
        neighbor_num_lst = self.neighbor_num_lst
        neighbor_id_lstlst = self.neighbor_id_lstlst


        pc_mask  = torch.ones(in_pn+1).float().to(in_pc.device)
        pc_mask[in_pn]=0
        neighbor_mask_lst = index_selection_nd(pc_mask,neighbor_id_lstlst,0).contiguous()#out_pn*max_neighbor_num neighbor is 1 otherwise 0

        raw_weights = self.weights
        bias = self.bias

        w_weights = raw_w_weights*(neighbor_mask_lst.view(out_pn, max_neighbor_num, 1)) #out_pn*max_neighbor_num*weight_num


        in_pc_pad = torch.cat((in_pc, torch.zeros(batch, 1, in_channel).float().to(in_pc.device)), 1) #batch (in_pn+1) in_channel
        in_neighbors = index_selection_nd(in_pc_pad,neighbor_id_lstlst, 1)
        fuse_neighbors = torch.einsum('pnm,bpni->bpmi',[w_weights, in_neighbors]) #batch*out_pn*max_neighbor_num*out_channel

        normalized_weights = raw_weights.view(weight_num,out_channel,in_channel)
        out_neighbors = torch.einsum('moi,bpmi->bpmo',[normalized_weights, fuse_neighbors]) #out_pn*max_neighbor_num*(out_channel*in_channel)


        out_pc = "" #batch*out_pn*out_channel
        if b_max_pool:
            out_pc = out_neighbors.max(2)
        else:
            out_pc = out_neighbors.sum(2)

        out_pc = out_pc + bias

        if is_final_layer==False:
            out_pc = self.relu(out_pc)
            # out_pc = self.relu(self.norm(out_pc.permute(0,2,1))).permute(0,2,1)
            # if self.dropout is not None:
            #     out_pc = self.relu(self.dropout(out_pc))
            # else:
            #     out_pc = self.relu(out_pc)

        if self.residual_rate==0:
            return out_pc

        if(in_channel != out_channel):
            in_pc_pad = torch.einsum('oi,bpi->bpo',[self.weight_res.view(out_channel,in_channel), in_pc_pad])

        out_pc_res = []
        if(in_pn == out_pn):
            out_pc_res = in_pc_pad[:,0:in_pn].clone()
        else:
            p_neighbors_raw = self.p_neighbors
            in_neighbors = index_selection_nd(in_pc_pad,neighbor_id_lstlst, 1)

            #p_neighbors = torch.sigmoid(p_neighbors_raw) * neighbor_mask_lst
            p_neighbors = torch.abs(p_neighbors_raw)  * neighbor_mask_lst
            p_neighbors_sum = p_neighbors.sum(1) + 1e-8 #out_pn
            p_neighbors = p_neighbors/p_neighbors_sum.view(out_pn,1).repeat(1,max_neighbor_num)

            out_pc_res = torch.einsum('pn,bpno->bpo', [p_neighbors, in_neighbors])

        out_pc = out_pc*np.sqrt(1-self.residual_rate) + out_pc_res*np.sqrt(self.residual_rate)

        return out_pc

class MCFixedEnc(nn.Module):
    def __init__(self,structure, channel_lst, weight_num): #layer_info_lst= [(point_num, feature_dim)]
        super(MCFixedEnc, self).__init__()

        self.point_num = structure.point_num
        self.residual_rate = structure.residual_rate
        self.b_max_pool = structure.b_max_pool
        self.perpoint_bias = structure.perpoint_bias
        self.channel_lst = channel_lst
        self.layer_num = len(structure.connection_info_lsts)
        self.layer_lst = nn.ModuleList([])

        b_Perpt_bias = self.perpoint_bias
        for l in np.arange(0,self.layer_num):
            in_channel = self.channel_lst[l]
            out_channel = self.channel_lst[l+1]
            connection_info  = structure.connection_info_lsts[l]
            in_point_num = structure.ptnum_list[l]
            self.layer_lst.append(LASMConvssw(in_channel, out_channel, weight_num,in_point_num,  connection_info, b_Perpt_bias, self.residual_rate))

        self.out_nrpts = structure.ptnum_list[self.layer_num]
        self.out_nrchs = out_channel

        # print(self.layer_num, self.out_nrpts, self.out_nrchs)

    def forward_till_layer_n(self,in_pc,vcoeffs, layer_n):
        out_pc = in_pc.clone()
        for i in range(layer_n):
            out_pc = self.layer_lst[i](out_pc,vcoeffs.vcoeffs_list[i], is_final_layer = False, b_max_pool = self.b_max_pool)
        return out_pc

    def forward(self, in_pc, vcoeffs):
        tmpcode = self.forward_till_layer_n(in_pc, vcoeffs, self.layer_num-1)
        out = self.layer_lst[self.layer_num-1](tmpcode, vcoeffs.vcoeffs_list[self.layer_num-1], is_final_layer = True, b_max_pool = self.b_max_pool)
        return out

class MCStructure(nn.Module):
    def __init__(self, channel_lst, connection_layer, inptnr, weight_num, bDec= True, b_perpoint_bias = True): #layer_info_lst= [(point_num, feature_dim)]
        super(MCStructure, self).__init__()

        self.point_num = inptnr

        self.residual_rate = 0.9    #param.residual_rate
        self.b_max_pool = 0         #param.conv_max
        self.perpoint_bias = b_perpoint_bias #param.perpoint_bias
        self.connection_folder = '../../MeshConvolution/flame/ConnectionMatrices/'

        fn_lst = os.listdir(self.connection_folder)
        connection_layer_fn_lst = []
        for layer_name in connection_layer:
            layer_name = "_"+layer_name+"."

            find_fn = False
            for fn in fn_lst:
                if((layer_name in fn) and ((".npy" in fn) or (".npz" in fn))):

                    connection_layer_fn_lst +=[self.connection_folder+fn]
                    find_fn = True
                    break
            if(find_fn ==False):
                print ("!!!ERROR: cannot find the connection layer fn")

        self.connection_layer_fn_lst = connection_layer_fn_lst
        self.layer_num = len(self.connection_layer_fn_lst)

        self.ptnum_list = []
        self.ptnum_list += [inptnr]
        self.connection_info_lsts = []
        for l in np.arange(0,self.layer_num):
            # print ("##Layer", self.connection_layer_fn_lst[l])
            connection_info  = np.load(self.connection_layer_fn_lst[l])
            out_point_num = connection_info.shape[0]
            self.connection_info_lsts += [connection_info]
            self.ptnum_list += [out_point_num]

    def forward(self):
        return

class MCVcoeffs(nn.Module):
    def __init__(self, structure, weight_num): #layer_info_lst= [(point_num, feature_dim)]
        super(MCVcoeffs, self).__init__()


        self.layer_num = len(structure.connection_layer_fn_lst)
        self.vcoeffs_list = nn.ParameterList([])
        for l in np.arange(0,self.layer_num):
            connection_info = structure.connection_info_lsts[l]
            out_point_num = connection_info.shape[0]

            neighbor_num_lst = torch.from_numpy(connection_info[:,0].astype(np.float32)).float() #out_point_num*1

            neighbor_id_dist_lstlst = connection_info[:, 1:] #out_point_num*(max_neighbor_num*2)
            neighbor_id_lstlst = neighbor_id_dist_lstlst.reshape((out_point_num, -1,2))[:,:,0] #out_point_num*max_neighbor_num

            neighbor_id_lstlst = torch.from_numpy(neighbor_id_lstlst).long()
            max_neighbor_num = neighbor_id_lstlst.shape[1]

            avg_neighbor_num= round(neighbor_num_lst.mean().item())

            w_weights=torch.randn(out_point_num, max_neighbor_num, weight_num)/(avg_neighbor_num*weight_num)
            w_weights = nn.Parameter(w_weights)
            self.vcoeffs_list.append(w_weights) #+= [w_weights]

class MeshConvDisc(nn.Module):
    def __init__(self, enc_channel_list, conn_layer_enc, point_num, weight_num):
        super(MeshConvDisc, self).__init__()

        self.enc_channel_list = enc_channel_list
        self.weight_num = weight_num
        self.point_num = point_num

        self.mcstructureenc = MCStructure(self.enc_channel_list, conn_layer_enc, self.point_num, self.weight_num, bDec=False)
        self.mcvcoeffsenc = MCVcoeffs(self.mcstructureenc, self.weight_num)
        self.enc = MCFixedEnc(self.mcstructureenc, self.enc_channel_list, self.weight_num)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.enc(x, self.mcvcoeffsenc)

        return out
