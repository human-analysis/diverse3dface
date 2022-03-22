import os, sys
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from glob import glob
import time
import datetime
import imageio
import pytorch3d
from pytorch3d import loss as _3d_losses
from pytorch3d.io import save_ply

sys.path.append('./flame_model/')
from FLAME import FLAME, FLAMETex
from renderer import SRenderY as Renderer
import utils
torch.backends.cudnn.benchmark = True

import graphVAESSW as vae_model
import graphAE_param_iso as meshconv_param
import random
from plyfile import PlyData

from unet_seg import UnetSeg
from hrnet import HighResolutionNet, hrnet_config
from collections import OrderedDict
import copy

def index_selection_nd(x, I, dim):
    target_shape = [*x.shape]
    del target_shape[dim]
    target_shape[dim:dim] = [*I.shape]
    return x.index_select(dim, I.view(-1)).reshape(target_shape)

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def threshold(x):
    x[x < 0.5] = 0
    x[x >= 0.5] = 1
    return x

def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds, maxval[:,:,0]


class Net_autoenc(nn.Module):
    def __init__(self, param, in_channels=3):
        super(Net_autoenc, self).__init__()
        self.weight_num = 17

        self.mcstructureenc = vae_model.MCStructure(param, param.point_num, self.weight_num, bDec =False)
        self.mcvcoeffsenc = vae_model.MCVcoeffs(self.mcstructureenc, self.weight_num)
        encgeochannel_list =  [in_channels,32,64,128,256,512,1024,64]
        self.net_geoenc = vae_model.MCEnc(self.mcstructureenc, encgeochannel_list, self.weight_num)

        self.nrpt_latent = self.net_geoenc.out_nrpts
        self.mcstructuredec = vae_model.MCStructure(param,self.nrpt_latent,self.weight_num, bDec = True)
        self.mcvcoeffsdec = vae_model.MCVcoeffs(self.mcstructuredec, self.weight_num)
        decgeochannel_list3 =  [64,1024,512,256,128,64,32,3]
        self.net_geodec = vae_model.MCDec(self.mcstructuredec, decgeochannel_list3, self.weight_num)

        self.net_loss = vae_model.MCLoss(param)
        # self.register_buffer('t_facedata', facedata.long())

    def forward(self, in_pc_batch): # meshvertices: B N 3, meshnormals: B N 3
        nbat = in_pc_batch.size(0)
        npt = in_pc_batch.size(1)
        nch = in_pc_batch.size(2)

        t_mu, t_logstd = self.net_geoenc(in_pc_batch, self.mcvcoeffsenc) # in in mm and out in dm
        t_std = t_logstd.exp()

        t_eps = torch.ones_like(t_std).normal_() #torch.FloatTensor(t_std.size()).normal_().to(device)
        t_z = t_mu + t_std * t_eps

        out_pc_batchfull = self.net_geodec(t_z, self.mcvcoeffsdec)
        out_pc_batch = out_pc_batchfull[:,:,0:3]

        return out_pc_batch, t_z, t_mu, t_std


class PhotometricFitting(object):
    def __init__(self, config, device='cuda'):
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.config = config
        self.device = device
        self.eye_pairs = ((37, 41), (38, 40), (43,47), (44, 46), (36, 39), (42, 45))
        self.mouth_pairs = ((61, 67), (62, 66), (63, 65), (60, 64))
        #
        self.flame = FLAME(self.config).to(self.device)
        self.flametex = FLAMETex(self.config).to(self.device)
        self.flame_mask = np.load('data/FLAME_masks.pkl', allow_pickle=True, encoding='latin1')

        # self.component_pca = np.load('./data/component_pca.npy', allow_pickle=True).item()              # n_shape=10, n_exp=10
        # self.component_se_pca = np.load('./data/component_se_pca.npy', allow_pickle=True).item()        # n_shape=10, n_exp=0
        self.component_se10_pca = np.load('./data/component_se10_pca.npy', allow_pickle=True).item()    # n_shape=10, n_exp=10
        self.custom_se10_pca = np.load('./data/component_se10_pca_custom.npy', allow_pickle=True).item()    # n_shape=10, n_exp=10
        self.custom_se10_pca['face'] = self.custom_se10_pca['cheeks']
        del self.custom_se10_pca['cheeks']
        self.model_types = ['component_se10_pca']

        for prop in ['shapedirs', 'mean', 'explained_variance']:
            for key in self.component_se10_pca.keys():
                for model_type in self.model_types:
                    getattr(self, model_type)[key][prop] = torch.tensor(getattr(self, model_type)[key][prop], dtype=torch.float).to(self.device)

        self.components = list(self.component_se10_pca.keys())
        self.ignore_regions = ['boundary', 'neck']
        for ignore_region in self.ignore_regions:
            for model_type in self.model_types:
                del getattr(self, model_type)[ignore_region]

        # pairwise weightage calculation
        self.pairwise_weights = {}
        self.template = self.flame.v_template
        self.pairwise_sigma_f = 16
        self.dist_template = torch.cdist(self.template, self.template).unsqueeze(0)
        sigma = self.dist_template.max()/self.pairwise_sigma_f
        for i in range(len(self.custom_se10_pca.keys())-1):
            key_i = list(self.custom_se10_pca.keys())[i]
            pairwise_weights = {}
            for j in range(i+1, len(self.custom_se10_pca.keys())):
                key_j = list(self.custom_se10_pca.keys())[j]
                v_key_i = self.custom_se10_pca[key_i]['indices']
                v_key_j = self.custom_se10_pca[key_j]['indices']
                dist_ij = self.dist_template[:,v_key_i][:,:,v_key_j]
                valid = (dist_ij < dist_ij.min()*10).float()
                sim_k = 0.25 / dist_ij.median()
                sim_ij = torch.exp(-dist_ij**2 / sigma**2)
                pairwise_weights[key_j] = sim_ij * valid
            self.pairwise_weights[key_i] = pairwise_weights

        # self.pairwise_weights = {}
        # self.template = self.flame.v_template
        # self.pairwise_sigma_f = 16
        # self.dist_template = torch.cdist(self.template, self.template).unsqueeze(0)
        # sigma = self.dist_template.max()/self.pairwise_sigma_f
        # self.pairwise_model = [['forehead', 'left_eye_region', 'right_eye_region', 'lips'], ['lips', 'nose', 'face']]
        # for pairs in self.pairwise_model:
        #     for i in range(len(pairs)-1):
        #         key_i = pairs[i]
        #         pairwise_weights = {}
        #         for j in range(i+1, len(pairs)):
        #             key_j = pairs[j]
        #             v_key_i = self.custom_se10_pca[key_i]['indices']
        #             v_key_j = self.custom_se10_pca[key_j]['indices']
        #             dist_ij = self.dist_template[:,v_key_i][:,:,v_key_j]
        #             sim_k = 0.25 / dist_ij.median()
        #             sim_ij = torch.exp(-dist_ij**2 / sigma**2)
        #             pairwise_weights[key_j] = sim_ij
        #         self.pairwise_weights[key_i] = pairwise_weights

        self._setup_renderer()

        # Meshconv configuration
        self.meshconv_param=meshconv_param.Parameters()
        self.meshconv_param.read_config(self.config.meshconv_config)
        self.in_channels = 3
        self.occ_input = 0
        # loading template mesh
        templydata = PlyData.read(self.meshconv_param.template_ply_fn)
        tri_idx = templydata['face']['vertex_indices']
        temply_facedata = torch.from_numpy(np.vstack(tri_idx))
        self.meshconv = Net_autoenc(self.meshconv_param, self.in_channels).to(self.device).eval()

        if(self.meshconv_param.read_weight_path!=""):
            print ("load "+self.meshconv_param.read_weight_path)
            checkpoint = torch.load(self.meshconv_param.read_weight_path)
            self.meshconv.net_geoenc.load_state_dict(checkpoint['encgeo_state_dict'])
            self.meshconv.net_geodec.load_state_dict(checkpoint['decgeo_state_dict'])
            self.meshconv.mcvcoeffsenc.load_state_dict(checkpoint['mcvcoeffsenc_dict'])
            self.meshconv.mcvcoeffsdec.load_state_dict(checkpoint['mcvcoeffsdec_dict'])

        self.num_samples = 16
        self.eye = torch.eye(self.num_samples).float().to(device)
        non_diag_idx = []
        for i in range(self.num_samples):
            for j in range(i+1, self.num_samples):
                non_diag_idx.append([i, j])
        self.non_diag_idx = torch.tensor(non_diag_idx).long().to(device)
        self.dpp_iter = config.dpp_iter

        self.w_nor = 10
        self.w_shape = 1000
        self.w_laplace = 1000
        self.w_pho = 500
        self.w_dpp = 0.025
        self.w_lmk = 10
        self._90_perc_radius = 20.9539

        # face parser
        self.unetseg = UnetSeg(in_channels=3, out_channels=3).to(self.device)
        unetseg_ckpt = torch.load('pretrained_models/checkpoint_occ_93.pth')
        self.unetseg.load_state_dict(unetseg_ckpt['modelO'])
        self.unetseg.eval()

        hrnet_config.defrost()
        hrnet_config.merge_from_file('./hrnet/hrnet_300w.yaml')
        hrnet_config.MODEL.INIT_WEIGHTS = False
        hrnet_config.freeze()
        self.hrnet = HighResolutionNet(hrnet_config).to(self.device)
        hrnet_state_dict = torch.load('pretrained_models/HR18-300W.pth')
        hrnet_state_dict_new = OrderedDict()
        for key in hrnet_state_dict.keys():
            new_key = key.replace('module.', '')
            hrnet_state_dict_new[new_key] = hrnet_state_dict[key]
        self.hrnet.load_state_dict(hrnet_state_dict_new)
        self.hrnet.eval()

        # flame regularizaiton loss
        lambda_reg = 0.001
        shapedirs = self.flame.shapedirs.view(5023*3, -1)
        mean_shape = self.flame.v_template
        XtX = shapedirs.T.matmul(shapedirs)
        XtX_inv = torch.inverse(XtX + lambda_reg)
        X_XtXinv = shapedirs.matmul(XtX_inv)
        self.X_XtXinv_Xt = X_XtXinv.matmul(shapedirs.T)

        self.interp = False

    def _setup_renderer(self):
        mesh_file = './data/head_template_mesh.obj'
        self.render = Renderer(self.image_size, obj_filename=mesh_file).to(self.device)

    def get_dpp_loss(self, shapes, mask=None, dsf_k=None, latent=None, mu=None, sigma=None, get_dsf=False):
        bs = shapes.shape[0]
        if len(shapes.shape) < 3:
            shapes = shapes.unsqueeze(1)
        num_f = shapes.shape[1]*3
        if num_f == 0:
            return torch.zeros([], device=shapes.device)

        if mask is not None:
            shapes = (shapes * mask) * torch.tensor(mask.shape).prod() / mask.sum()
        shapes = shapes.view(-1, self.num_samples, num_f).contiguous()
        distances = torch.cdist(shapes, shapes).mean(0) / np.sqrt(num_f//3)

        if dsf_k is None or dsf_k == 1:
            # median of the upper triangle distances matrix D except the diagonal / (0.5 * D.median()), may be change 0.5 to sth else
            dsf_k = 0.5 / distances[self.non_diag_idx[:,0], self.non_diag_idx[:,1]].median().detach()

        similarity = torch.exp(-dsf_k * distances)

        latent = latent.view(bs, self.num_samples, -1)
        latent_f = latent.shape[-1]
        latent_norm = latent.norm(dim=2).mean(0)
        exceeds = torch.max(latent_norm - self._90_perc_radius, torch.zeros_like(latent_norm))
        quality = torch.exp(-exceeds)

        kernel = torch.pow(quality, 2) * similarity
        dpp_kernel_inv = torch.inverse(kernel + self.eye)
        loss_dpp = - torch.trace(self.eye - dpp_kernel_inv)*2

        if get_dsf:
            return loss_dpp, dsf_k, distances
        return loss_dpp

    # def pairwise_loss(self, shapes):
    #     pairwise_loss = 0
    #     bs = shapes.shape[0]
    #     n = 0
    #     self_dist = torch.cdist(shapes, shapes)

    #     for pairs in self.pairwise_model:
    #         for i in range(len(pairs)-1):
    #             key_i = pairs[i]
    #             for j in range(i+1, len(pairs)):
    #                 key_j = pairs[j]
    #                 v_key_i = self.custom_se10_pca[key_i]['indices']
    #                 v_key_j = self.custom_se10_pca[key_j]['indices']
    #                 self_dist_ij = self_dist[:,v_key_i][:,:,v_key_j]
    #                 template_dist_ij = self.dist_template[:,v_key_i][:,:,v_key_j]
    #                 pair_loss = self.pairwise_weights[key_i][key_j] * torch.abs(self_dist_ij - template_dist_ij)
    #                 pairwise_loss += pair_loss.sum((1,2))
    #                 n += 1

    #     return (pairwise_loss / n).mean()

    def pairwise_loss(self, shapes):
        pairwise_loss = 0
        bs = shapes.shape[0]
        n = len(self.custom_se10_pca.keys())
        self_dist = torch.cdist(shapes, shapes)
        for i in range(n-1):
            key_i = list(self.custom_se10_pca.keys())[i]
            for j in range(i+1, n):
                key_j = list(self.custom_se10_pca.keys())[j]
                # if key_i == 'lips' or key_j == 'lips':
                v_key_i = self.custom_se10_pca[key_i]['indices']
                v_key_j = self.custom_se10_pca[key_j]['indices']
                self_dist_ij = self_dist[:,v_key_i][:,:,v_key_j]
                template_dist_ij = self.dist_template[:,v_key_i][:,:,v_key_j]
                pair_loss = self.pairwise_weights[key_i][key_j] * torch.abs(self_dist_ij - template_dist_ij)
                pairwise_loss += pair_loss.sum((1,2))

        # return (pairwise_loss / (n*(n-1)/2)).mean()
        return (pairwise_loss / n).mean()


    def eye_loss(self, lmk_in, lmk_out, lmk_conf=None):
        if lmk_conf is None:
            lmk_conf = torch.ones_like(lmk_in)[:,:,0]
        loss = 0
        for pair in self.eye_pairs:
            loss += torch.norm((lmk_in[:,pair[0]] - lmk_in[:,pair[1]]) - (lmk_out[:,pair[0]] - lmk_out[:,pair[1]]), dim=1) * lmk_conf[:,pair[0]] * lmk_conf[:,pair[1]]
        loss = loss.mean()
        return loss

    def mouth_loss(self, lmk_in, lmk_out, lmk_conf=None):
        if lmk_conf is None:
            lmk_conf = torch.ones_like(lmk_in)[:,:,0]
        loss = 0
        for pair in self.mouth_pairs:
            loss += torch.norm((lmk_in[:,pair[0]] - lmk_in[:,pair[1]]) - (lmk_out[:,pair[0]] - lmk_out[:,pair[1]]), dim=1) * lmk_conf[:,pair[0]] * lmk_conf[:,pair[1]]
        loss = loss.mean()
        return loss

    def fit_visible(self, images, landmarks, image_masks, occ_dict, savefolder=None, num_steps=2000, n_shape=None, n_exp=None, component_pca_type=None, shape_gt=None):
        bz = images.shape[0]
        if n_shape is None:
            n_shape = self.config.shape_params
        if n_exp is None:
            n_exp = self.config.expression_params
        image_size = images.shape[-1]
        images = images.clone()
        image_masks = image_masks.clone()
        faces = self.flame.faces_tensor.unsqueeze(0).expand(bz,-1,-1)
        gt_landmark = landmarks

        occ_mask = occ_dict['occ_mask']
        landmarks_valid = occ_dict['landmarks_valid']
        landmarks_conf = occ_dict['landmarks_conf']
        image_masks[occ_mask==1] = 0
        images[occ_mask.expand(-1,3,-1,-1)==1] = 0

        shape = nn.Parameter(torch.zeros(bz, n_shape).float().to(self.device))
        tex = nn.Parameter(torch.zeros(bz, self.config.tex_params).float().to(self.device))
        exp = nn.Parameter(torch.zeros(bz, n_exp).float().to(self.device))
        pose = nn.Parameter(torch.zeros(bz, self.config.pose_params).float().to(self.device))
        eye_pose = nn.Parameter(torch.zeros(bz, 6).float().to(self.device))
        neck_pose = nn.Parameter(torch.zeros(bz, 3).float().to(self.device))
        cam = torch.zeros(bz, self.config.camera_params); cam[:, 0] = 5.
        cam = nn.Parameter(cam.float().to(self.device))
        lights = nn.Parameter(torch.zeros(bz, 9, 3).float().to(self.device))

        opt_rigid_list = [pose, eye_pose, neck_pose, cam]
        opt_list = [shape, exp, tex, lights, pose, eye_pose, neck_pose, cam]

        if component_pca_type is not None:
            component_pca = getattr(self, component_pca_type)
            component_params = {}
            for key in component_pca.keys():
                n_comp = component_pca[key]['shapedirs'].shape[-1]
                betas = nn.Parameter(torch.normal(torch.zeros(bz, n_comp), torch.ones(bz, n_comp)).float().to(self.device))
                component_params[key] = betas
                opt_list.append(component_params[key])
            component_pca_config = {'model': component_pca, 'params': component_params, 'n_baseshape': n_shape, 'n_baseexp': n_exp}
            model_name = component_pca_type
        else:
            component_pca_config = None
            component_params = None
            model_name = 'full'
        # utils.check_mkdir(os.path.join(savefolder, model_name))
        utils.check_mkdir(os.path.join(savefolder, 'iterations'))

        e_opt_rigid = torch.optim.Adam(
            opt_rigid_list,
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )
        e_opt = torch.optim.Adam(
            opt_list,
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )

        # rigid fitting of pose and camera with 51 static face landmarks,
        # this is due to the non-differentiable attribute of contour landmarks trajectory
        visind = range(min(bz, 8))
        for k in range(200):
            losses = {}
            vertices, landmarks2d, landmarks3d, full_shape = self.flame(shape_params=shape, expression_params=exp, pose_params=pose, eye_pose_params=eye_pose,
                                                                        neck_pose_params=neck_pose, component_pca_config=component_pca_config)
            trans_vertices = utils.batch_orth_proj(vertices, cam);
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = utils.batch_orth_proj(landmarks2d, cam);
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = utils.batch_orth_proj(landmarks3d, cam);
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['lmk'] = utils.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2], conf=landmarks_valid[:,17:]) * config.w_lmks
            losses['pose_reg'] = (torch.sum(pose ** 2) / 2 + torch.sum(eye_pose ** 2) / 2 + torch.sum(neck_pose ** 2) / 2) * config.w_pose_reg

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt_rigid.zero_grad()
            all_loss.backward()
            e_opt_rigid.step()

            loss_info = '----iter: {}\t'.format(k)
            for key in losses.keys():
                loss_info = loss_info + '{}: {:.4e}, '.format(key, float(losses[key]))
            if k % 50 == 0:
                print(loss_info)

            # if k % 100 == 0:
            #     grids = {}
            #     grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
            #     grids['landmarks_gt'] = torchvision.utils.make_grid(
            #         utils.tensor_vis_landmarks(images[visind], landmarks[visind], valid=landmarks_valid[visind]))
            #     grids['landmarks2d'] = torchvision.utils.make_grid(
            #         utils.tensor_vis_landmarks(images[visind], landmarks2d[visind], valid=landmarks_valid[visind]))
            #     grids['landmarks3d'] = torchvision.utils.make_grid(
            #         utils.tensor_vis_landmarks(images[visind], landmarks3d[visind], valid=landmarks_valid[visind]))

            #     grid = torch.cat(list(grids.values()), 1)
            #     grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
            #     grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
                # cv2.imwrite('{}/{}/{}.jpg'.format(savefolder, model_name, k), grid_image)

        # non-rigid fitting of all the parameters with 68 face landmarks, photometric loss and regularization terms.
        for k in range(200, config.num_steps):
            losses = {}
            vertices, landmarks2d, landmarks3d, full_shape = self.flame(shape_params=shape, expression_params=exp, pose_params=pose, eye_pose_params=eye_pose,
                                                                        neck_pose_params=neck_pose, component_pca_config=component_pca_config)
            trans_vertices = utils.batch_orth_proj(vertices, cam);
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = utils.batch_orth_proj(landmarks2d, cam);
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = utils.batch_orth_proj(landmarks3d, cam);
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            # if is_occ:
            #     occ_verts_idx = (trans_vertices[:,:,0] > occ_x_start) * (trans_vertices[:,:,0] < occ_x_end) * (trans_vertices[:,:,1] > occ_y_start) * (trans_vertices[:,:,1] < occ_y_end)
            #     vis_verts_idx = ~occ_verts_idx

            losses['lmk'] = utils.l2_distance(landmarks2d[:, :, :2], gt_landmark[:, :, :2], conf=landmarks_valid) * config.w_lmks
            shape_reg_loss = (torch.sum(shape ** 2) / 2) * config.w_shape_reg
            exp_reg_loss = (torch.sum(exp ** 2) / 2) * config.w_expr_reg
            pose_reg_loss = (torch.sum(pose ** 2) / 2 + torch.sum(eye_pose ** 2) / 2 + torch.sum(neck_pose ** 2) / 2) * config.w_pose_reg
            losses['reg'] = shape_reg_loss + exp_reg_loss + pose_reg_loss
            if component_pca_type is not None:
                component_reg = 0
                for key in component_params.keys():
                    component_reg += torch.sum(component_params[key] ** 2) / 2
                component_reg_loss = component_reg * config.w_expr_reg
                losses['reg'] += component_reg_loss
            mouth_loss = self.mouth_loss(gt_landmark[:, :, :2], landmarks2d[:, :, :2], landmarks_valid)
            eye_loss = self.eye_loss(gt_landmark[:, :, :2], landmarks2d[:, :, :2], landmarks_valid)
            losses['mouth_eye'] = mouth_loss + eye_loss

            # res_shape = (full_shape - self.flame.v_template).view(-1, 1)
            # losses['resreg'] = (res_shape - self.X_XtXinv_Xt.matmul(res_shape)).view(5023,3).norm(dim=1).mean()*1e7
            # losses['pairwise'] = self.pairwise_loss(full_shape)*10

            meshes = pytorch3d.structures.Meshes(full_shape, faces)
            losses['smooth'] = _3d_losses.mesh_laplacian_smoothing(meshes)*config.w_smooth

            ## render
            albedos = self.flametex(tex) / 255.
            ops = self.render(vertices, trans_vertices, albedos, lights)
            predicted_images = ops['images']
            losses['pho'] = (image_masks * (ops['images'] - images).abs()).mean() * config.w_pho

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt.zero_grad()
            all_loss.backward()
            e_opt.step()

            loss_info = '----iter: {}\t'.format(k)
            for key in losses.keys():
                loss_info = loss_info + '{}: {:.4e}, '.format(key, float(losses[key]))

            if k % 50 == 0:
                print(loss_info)

            # visualize
            if k % 200 == 0:
                grids = {}
                visind = range(bz)  # [0]
                grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                grids['landmarks_gt'] = torchvision.utils.make_grid(
                    utils.tensor_vis_landmarks(images[visind], landmarks[visind], valid=landmarks_valid[visind]))
                grids['landmarks2d'] = torchvision.utils.make_grid(
                    utils.tensor_vis_landmarks(images[visind], landmarks2d[visind], valid=landmarks_valid[visind]))
                grids['landmarks3d'] = torchvision.utils.make_grid(
                    utils.tensor_vis_landmarks(images[visind], landmarks3d[visind], valid=landmarks_valid[visind]))
                grids['albedoimage'] = torchvision.utils.make_grid(
                    (ops['albedo_images'])[visind].detach().cpu())
                grids['render'] = torchvision.utils.make_grid(predicted_images[visind].detach().float().cpu())
                shape_images = self.render.render_shape(vertices, trans_vertices)
                grids['shape'] = torchvision.utils.make_grid(
                    F.interpolate(shape_images[visind], [224, 224])).detach().float().cpu()

                # grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos[visind], [224, 224])).detach().cpu()
                grid = torch.cat(list(grids.values()), 1)
                grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
                cv2.imwrite('{}/iterations/{}.jpg'.format(savefolder,  k), grid_image)

        occ_vt = self.render.image2verts(trans_vertices, occ_mask).detach()
        occ_verts_idx = occ_vt[:,:,0].bool()
        vis_verts_idx = ~ occ_verts_idx

        full_pose = torch.cat([pose[:, :3], neck_pose, pose[:, 3:], eye_pose], dim=1)
        fitted_params = {
            'shape': vertices.detach(),
            'shape_images': shape_images.detach(),
            'vis_mask': vis_verts_idx,
            'exp': exp.detach(),
            'pose': pose.detach(),
            'full_pose': full_pose.detach(),
            'cam': cam.detach(),
            'albedo':albedos.detach(),
            'tex': tex.detach(),
            'light': lights.detach(),
            'component_params': component_params if component_pca_type is not None else None
        }
        # mesh = Mesh(v=vertices[0].detach().cpu().numpy(), f=faces[0].cpu().numpy(), vc=occ_vt[0].expand(-1,3).detach().cpu().numpy())
        # mesh.write_ply('{}/fitted.ply'.format(savefolder))
        save_ply('{}/fitted.ply'.format(savefolder), verts=vertices[0].detach().cpu(), faces=faces[0].cpu())
        # return single_params, shape_images
        # shape_error = (vertices[0] - shape_gt).norm() / np.sqrt(5023) * 1000
        # print('{} fitting shape error: {}'.format(model_name, shape_error))

        return fitted_params

    def diversify(self, fitted_params, images, landmarks, image_masks, occ_dict, savefolder, model_name='full', shape_gt=None):
        shape_in = fitted_params['shape'].expand(self.num_samples,-1,-1)
        cam = fitted_params['cam']
        albedo = fitted_params['albedo'].expand(self.num_samples,-1,-1,-1)
        light = fitted_params['light'].expand(self.num_samples,-1,-1)
        pose = fitted_params['full_pose'].expand(self.num_samples,-1,-1)
        images = images.clone().expand(self.num_samples,-1,-1,-1)
        images_occ = images * image_masks
        batch_size = shape_in.shape[0]
        gt_landmarks = landmarks
        if shape_gt is not None:
            shape_gt = shape_gt.unsqueeze(0).expand(self.num_samples,-1,-1)
        else:
            shape_gt = shape_in

        # occlusion
        occ = occ_dict['occ_mask']
        landmarks_valid = occ_dict['landmarks_valid']
        landmarks_conf = occ_dict['landmarks_conf']
        image_masks[occ==1] = 0
        images[occ.expand(self.num_samples,3,-1,-1)==1] = 0
        images = images.expand(self.num_samples,-1,-1,-1)
        images_occ = images * image_masks

        # remove eye-balls
        vis_mask = fitted_params['vis_mask'].nonzero()[:,1].cpu().numpy()
        occ_mask = np.setdiff1d(np.arange(0, 5023), vis_mask)
        orig_occ_mask = copy.deepcopy(occ_mask)
        occ_mask = np.setdiff1d(occ_mask, self.flame_mask['left_eyeball'])
        occ_mask = np.setdiff1d(occ_mask, self.flame_mask['right_eyeball'])
        vis_mask = np.setdiff1d(np.arange(0, 5023), occ_mask)

        vis_mask_in = fitted_params['vis_mask'].unsqueeze(-1).float()
        vis_mask_in[:,vis_mask] = 1
        occ_mask_in = 1 - vis_mask_in

        _, _, landmarks_vis = self.flame.get_landmarks_vis(shape_in[0:1], pose[0:1], vis_mask_in.repeat(1,1,3))
        landmarks_vis = threshold(landmarks_vis[:,:,0]).long().detach()
        landmarks_valid *= landmarks_vis

        model_in = shape_in.clone()
        model_in_vis = shape_in.clone()
        model_in[:,occ_mask] = 0
        model_in_vis[:,orig_occ_mask] = 0

        with torch.no_grad():
            shape_out, z, mu, std = self.meshconv(model_in)

        z.requires_grad = True
        optimizer = torch.optim.Adam([z], lr=1e-2, weight_decay=5e-4)
        utils.check_mkdir(os.path.join(savefolder, model_name))

        gt_trans_verts = utils.batch_orth_proj(shape_in, cam)
        gt_trans_verts[:,:,1:] = - gt_trans_verts[:,:,1:]
        shape_images_in = self.render.render_shape(model_in_vis, gt_trans_verts)

        for iter in range(self.dpp_iter):
            optimizer.zero_grad()
            self.meshconv.zero_grad()

            shape_out = self.meshconv.net_geodec(z, self.meshconv.mcvcoeffsdec)
            shape_loss = self.meshconv.net_loss.compute_geometric_loss_l1(shape_in[0:1,vis_mask,0:3], shape_out[:,vis_mask,0:3])
            laplace_loss = self.meshconv.net_loss.compute_laplace_loss_l2(shape_in * vis_mask_in, shape_out * vis_mask_in)
            dpp_loss = self.get_dpp_loss(shape_out[:, occ_mask], latent=z)

            # other losses
            landmarks2d, landmarks3d = self.flame.get_landmarks(shape_out, pose)
            trans_verts = utils.batch_orth_proj(shape_out, cam)
            trans_verts[:,:,1:] = - trans_verts[:,:,1:]
            landmarks2d = utils.batch_orth_proj(landmarks2d, cam);
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = utils.batch_orth_proj(landmarks3d, cam);
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            lmk_loss = utils.l2_distance(landmarks2d[:, :, :2], gt_landmarks[:, :, :2], conf=landmarks_valid)*0
            mouth_loss = self.mouth_loss(gt_landmarks[:, :, :2], landmarks2d[:, :, :2], landmarks_valid)*0
            eye_loss = self.eye_loss(gt_landmarks[:, :, :2], landmarks2d[:, :, :2], landmarks_valid)*0
            mouth_eye_loss = mouth_loss + eye_loss

            ops = self.render(shape_out, trans_verts, albedo, light)
            predicted_images = ops['images']
            pho_loss = (image_masks[0:1] * (ops['images'] - images[0:1]).abs()).mean()

            loss = shape_loss * self.w_shape + laplace_loss * self.w_laplace + pho_loss * self.w_pho + lmk_loss * self.w_lmk + mouth_eye_loss * self.w_lmk + dpp_loss * self.w_dpp
            loss.backward()
            optimizer.step()

            if (iter+1) % 100 == 0:
                shape_out = shape_out.detach()
                act_distance = torch.cdist(shape_out.view(batch_size,-1), shape_out.view(batch_size,-1)) / np.sqrt(5023)
                distances_vis = torch.cdist(shape_out[:,vis_mask].view(batch_size,-1), shape_out[:,vis_mask].view(batch_size,-1)) / np.sqrt(len(vis_mask))
                distances_occ = torch.cdist(shape_out[:,occ_mask].view(batch_size,-1), shape_out[:,occ_mask].view(batch_size,-1)) / np.sqrt(len(occ_mask))

                asd = (act_distance + 1000 * self.eye).min(dim=1)[0].mean() * 1000
                asd_vis = (distances_vis + 1000 * self.eye).min(dim=1)[0].mean() * 1000
                asd_occ = (distances_occ + 1000 * self.eye).min(dim=1)[0].mean() * 1000
                cse = (shape_out - shape_gt).norm(dim=(1,2)).min().item() / np.sqrt(5023) * 1000

                shape_images_out = self.render.render_shape(shape_out.detach(), trans_verts.detach())
                grid_image = torchvision.utils.make_grid(torch.cat((images, images_occ, shape_images_in, shape_images_out), dim=0).float().cpu(), nrow=self.num_samples)
                grid_image = (grid_image.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

                if iter < self.dpp_iter-1:
                    metrics_output = "{}/{} shape: {:.4e}, pho: {:.4e}, lmk: {:.4e}, dpp: {:.4e}, AED: {:.3f}, ASD: {:.3f}, ASD_VIS: {:.3f}, ASD_OCC: {:.3f}".format(iter, self.dpp_iter, shape_loss, pho_loss, lmk_loss, dpp_loss, cse, asd, asd_vis, asd_occ)
                    print(metrics_output)
                    cv2.imwrite('{}/iterations/diverse_{}.jpg'.format(savefolder, iter), grid_image)
                else:
                    metrics_output = "Ours: CSE: {:.4f}, ASD: {:.4f}, ASD_VIS: {:.4f}, ASD_OCC: {:.4f}".format(cse, asd, asd_vis, asd_occ)
                    print(metrics_output)
                    # eval_metrics.write(metrics_output + "\n")
                    cv2.imwrite('{}/{}.jpg'.format(savefolder, 'output_grid'), grid_image)

        for j in range(shape_out.shape[0]):
            # mesh = Mesh(v=shape_out[j].detach().cpu().numpy(), f=self.flame.faces_tensor.cpu().numpy())
            # mesh.write_ply('{}/shape_{}.ply'.format(savefolder, j))
            save_ply('{}/shape_{}.ply'.format(savefolder, j), verts=shape_out[j].detach().cpu(), faces=self.flame.faces_tensor.cpu())

        return shape_out.detach()

    def run(self, imagepath, occmaskpath, shapegt_path):
        # The implementation is potentially able to optimize with images(batch_size>1),
        # here we show the example with a single image fitting
        images = []
        occ_masks = []
        landmarks = []
        image_masks = []

        image_name = os.path.basename(imagepath)[:-4]
        savefile = os.path.sep.join([self.config.savefolder, image_name + '.npy'])

        # photometric optimization is sensitive to the hair or glass occlusions,
        # therefore we use a face segmentation network to mask the skin region out.

        image = cv2.resize(cv2.imread(imagepath), (config.cropped_size, config.cropped_size)).astype(np.float32) / 255.
        image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
        images.append(torch.from_numpy(image[None, :, :, :]).to(self.device))
        images = torch.cat(images, dim=0)
        images = F.interpolate(images, [self.image_size, self.image_size])

        if occmaskpath != '':
            occ_mask = cv2.resize(cv2.imread(occmaskpath), (config.cropped_size, config.cropped_size)).astype(np.float32) / 255.
            occ_mask = occ_mask.transpose(2, 0, 1)[0:1]
            occ_masks.append(torch.from_numpy(occ_mask[None, :, :, :]).to(self.device))
            occ_masks = torch.cat(occ_masks, dim=0)
            occ_masks = threshold(F.interpolate(occ_masks, [self.image_size, self.image_size]))
            occ_dict = {'occ_mask': occ_masks}

        try:
            image_mask = np.load(image_mask_path, allow_pickle=True)
            image_mask = image_mask[..., None].astype('float32')
            image_mask = image_mask.transpose(2, 0, 1)
            image_mask_bn = np.zeros_like(image_mask)
            image_mask_bn[np.where(image_mask != 0)] = 1.
            image_masks.append(torch.from_numpy(image_mask_bn[None, :, :, :]).to(self.device))
            image_masks = torch.cat(image_masks, dim=0)
            image_masks = F.interpolate(image_masks, [self.image_size, self.image_size])
        except:
            with torch.no_grad():
                face_parsing_out = self.unetseg(images*2-1)
                # face_mask = threshold(face_parsing_out[:,0:1])
                background_mask = threshold(face_parsing_out[:,2:3])
                image_masks = threshold(face_parsing_out[:,0:1])

                if occmaskpath == '':
                    occ_masks = threshold(face_parsing_out[:,1:2])
                    occ_dict = {'occ_mask': occ_masks}

        try:
            landmark = np.load(landmarkpath).astype(np.float32)
            landmark[:, 0] = landmark[:, 0] / float(image.shape[2]) * 2 - 1
            landmark[:, 1] = landmark[:, 1] / float(image.shape[1]) * 2 - 1
            landmarks.append(torch.from_numpy(landmark)[None, :, :].float().to(self.device))
            landmarks = torch.cat(landmarks, dim=0)

        except:
            with torch.no_grad():
                images_up = nn.functional.interpolate(images, size=(self.image_size, self.image_size)) * 2 - 1
                lmark_hm = self.hrnet(images_up)
                landmarks, landmarks_conf = get_preds(lmark_hm)
                landmarks /= self.image_size // 4
                landmarks = landmarks*2 - 1
                occ_dict['landmarks_valid'] = (landmarks_conf > 0.2).long()
                occ_dict['landmarks_conf'] = normalize(landmarks_conf)**2

        savefolder = os.path.sep.join([self.config.savefolder, '{}'.format(image_name)])
        utils.check_mkdir(savefolder)

        # shape_gt = Mesh(filename=shapegt_path)
        # shape_gt.write_ply('{}/groundtruth.ply'.format(savefolder))
        # shape_gt = torch.tensor(shape_gt.v).to(self.device)
        shape_gt=None

        # optimize
        for pca_config in [(10, 10, 'component_se10_pca')]:#, (10, 10, 'custom_se10_pca')]:
            print(savefolder, pca_config[2])
            fitted_params = self.fit_visible(images, landmarks, image_masks, occ_dict, savefolder, num_steps=self.config.num_steps, n_shape=pca_config[0], n_exp=pca_config[1], component_pca_type=pca_config[2], shape_gt=shape_gt)
            self.diversify(fitted_params, images, landmarks, image_masks, occ_dict, savefolder, pca_config[2], shape_gt=shape_gt)


if __name__ == '__main__':
    if len(sys.argv) == 5:
        image_path = str(sys.argv[1])
        mask_path = str(sys.argv[2])
        meshconv_config = str(sys.argv[3])
        savefolder = str(sys.argv[4])
    else:
        image_path = str(sys.argv[1])
        mask_path = ''
        meshconv_config = str(sys.argv[2])
        savefolder = str(sys.argv[3])
    shapegt_path = ''
    device_name = 'cuda'
    config = {
        # FLAME
        'flame_model_path': './data/generic_model.pkl',  # acquire it from FLAME project page
        'flame_lmk_embedding_path': './data/landmark_embedding.npy',
        'tex_space_path': './data/FLAME_texture.npz',  # acquire it from FLAME project page
        'camera_params': 3,
        'shape_params': 300,
        'expression_params': 100,
        'pose_params': 6,
        'tex_params': 50,
        'use_face_contour': True,

        'cropped_size': 256,
        'batch_size': 1,
        'image_size': 224,
        'e_lr': 0.005,
        'e_wd': 0.0001,
        'savefolder': savefolder,

        # weights of fitting losses and reg terms
        'w_pho': 16,
        'w_lmks': 5,
        'w_shape_reg': 1e-2,
        'w_expr_reg': 1e-2,
        'w_pose_reg': 1e-2,
        'w_smooth': 0.1,
        'num_steps': 1000,

        # dpp params
        'dpp_iter': 300,
        'dsf_k': 0.25,

        'meshconv_config': meshconv_config
    }

    config = utils.dict2obj(config)
    utils.check_mkdir(config.savefolder)

    config.batch_size = 1
    fitting = PhotometricFitting(config, device=device_name)

    fitting.run(image_path, mask_path, shapegt_path)
