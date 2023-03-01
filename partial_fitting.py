import os, sys
import cv2
import torch
import torchvision
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pytorch3d
from pytorch3d import loss as _3d_losses
from pytorch3d.io import save_ply
# from pytorch3d import Meshes, TexturesUV
from psbody.mesh import Mesh
import dnnlib

cur_dir_path = os.path.dirname(__file__)
os.chdir(cur_dir_path)

# sys.path.append('./flame_model/')
from flame_model.FLAME import FLAME, FLAMETex
from renderer import SRenderY as Renderer
import utils
torch.backends.cudnn.benchmark = True

import graphVAESSW as vae_model
import graphAE_param_iso as meshconv_param
from plyfile import PlyData

from unet_seg import UnetSeg
from hrnet import HighResolutionNet, hrnet_config
import face_alignment
from collections import OrderedDict
import copy
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class PhotometricFitting(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.config = args
        self.device = torch.device(args.device)
        self.eye_pairs = ((37, 41), (38, 40), (43,47), (44, 46), (36, 39), (42, 45))
        self.mouth_pairs = ((61, 67), (62, 66), (63, 65), (60, 64))
        #
        self.flame = FLAME(args).to(self.device)
        self.flametex = FLAMETex(args).to(self.device)
        self.flame_mask = np.load('data/FLAME_masks.pkl', allow_pickle=True, encoding='latin1')

        # self.component_pca = np.load('./data/component_pca.npy', allow_pickle=True).item()              # n_shape=10, n_exp=10
        # self.component_se_pca = np.load('./data/component_se_pca.npy', allow_pickle=True).item()        # n_shape=10, n_exp=0
        self.component_se10_pca = np.load('./data/component_se10_pca.npy', allow_pickle=True).item()    # n_shape=10, n_exp=10
        self.custom_se10_pca = np.load('./data/component_se10_pca_custom.npy', allow_pickle=True).item()    # n_shape=10, n_exp=10
        self.custom_se10_pca['face'] = self.custom_se10_pca['cheeks']
        del self.custom_se10_pca['cheeks']
        self.model_types = ['component_se10_pca', 'custom_se10_pca']

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

        self._setup_renderer()

        # face parser
        self.unetseg = UnetSeg(in_channels=3, out_channels=3).to(self.device)
        unetseg_ckpt = torch.load('pretrained_models/checkpoint_occ_93.pth')
        self.unetseg.load_state_dict(unetseg_ckpt['modelO'])
        self.unetseg.eval()

        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._3D, device='cuda')
        self.fa_2d = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, device='cuda')
        self.detected_faces = np.array([[[int(self.image_size * 0.15), int(self.image_size * 0.15), int(
            self.image_size * 0.85)-1, int(self.image_size * 0.85)-1, 1]]] * self.batch_size)

        # flame regularizaiton loss
        lambda_reg = 0.001
        shapedirs = self.flame.shapedirs.view(5023*3, -1)
        mean_shape = self.flame.v_template
        XtX = shapedirs.T.matmul(shapedirs)
        XtX_inv = torch.inverse(XtX + lambda_reg)
        X_XtXinv = shapedirs.matmul(XtX_inv)
        self.X_XtXinv_Xt = X_XtXinv.matmul(shapedirs.T)

        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url, cache_dir='pretrained') as f:
            self.vgg16 = torch.jit.load(f).eval().to(self.device)
        requires_grad(self.vgg16, False)

        self.interp = False

    def _setup_renderer(self):
        mesh_file = './data/head_template_mesh.obj'
        self.render = Renderer(self.image_size, obj_filename=mesh_file).to(self.device)

        template_mesh = Mesh(filename=mesh_file)
        self.vt = template_mesh.vt
        self.ft = template_mesh.ft


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

    def fit_visible(self, images, landmarks, image_masks, occ_dict, savefolder=None, n_shape=None, n_exp=None, component_pca_type=None, shape_gt=None):
        bz = images.shape[0]
        faces = self.flame.faces_tensor.unsqueeze(0).expand(bz,-1,-1)
        gt_landmark = landmarks

        writer = SummaryWriter(savefolder)
        writer.add_images('input', normalize(images))
        writer.add_images('facemask', image_masks)

        images = images.clone()
        image_masks = image_masks.clone()
        occ_mask = occ_dict['occ_mask']
        landmarks_valid = occ_dict['landmarks_valid']
        landmarks_conf = occ_dict['landmarks_conf']
        image_masks[occ_mask==1] = 0
        images[occ_mask.expand(-1, 3, -1, -1) == 1] = 0

        writer.add_images('occmask', occ_mask)
        writer.add_images('masked_image', images)

        # # visualize landmarks
        images_vis = nn.functional.interpolate(
            images, size=(self.image_size, self.image_size)) * 2 - 1
        for i in range(68):
            if (landmarks_valid[0, i]):
                images_vis[0, :, landmarks[0, i, 1].long()-1:landmarks[0, i, 1].long()+2, landmarks[0, i, 0].long(
                )-1:landmarks[0, i, 0].long()+2] = torch.tensor([0.0, 0.0, 1.0]).cuda()[:, None, None]
            else:
                images_vis[0, :, landmarks[0, i, 1].long()-1:landmarks[0, i, 1].long()+2, landmarks[0, i, 0].long(
                )-1:landmarks[0, i, 0].long()+2] = torch.tensor([1.0, 0.0, 0.0]).cuda()[:, None, None]
        writer.add_images('landmarks', images_vis)

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

        e_opt_rigid = torch.optim.AdamW(opt_rigid_list, lr=self.config.lr, weight_decay=self.config.weight_decay)
        e_opt = torch.optim.AdamW(opt_list, lr=self.config.lr, weight_decay=self.config.weight_decay)

        # rigid fitting of pose and camera with 51 static face landmarks,
        # this is due to the non-differentiable attribute of contour landmarks trajectory
        visind = range(min(bz, 8))
        pbar = tqdm(range(200))
        for k in pbar:
            losses = {}
            vertices, landmarks2d, landmarks3d, unposed_vertices = self.flame(shape_params=shape, expression_params=exp, pose_params=pose, eye_pose_params=eye_pose,
                                                                        neck_pose_params=neck_pose, component_pca_config=component_pca_config)
            trans_vertices = utils.batch_orth_proj(vertices, cam);
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = utils.batch_orth_proj(landmarks2d, cam);
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = utils.batch_orth_proj(landmarks3d, cam);
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['lmk'] = utils.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2], conf=landmarks_valid[:,17:]) * self.config.w_lmk
            losses['pose_reg'] = (torch.sum(pose ** 2) / 2 + torch.sum(eye_pose ** 2) / 2 + torch.sum(neck_pose ** 2) / 2) * self.config.w_shape_reg

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt_rigid.zero_grad()
            all_loss.backward()
            e_opt_rigid.step()

            # loss_info = '----iter: {}\t'.format(k)
            loss_info = ''
            for key in losses.keys():
                loss_info = loss_info + '{}: {:.4e}, '.format(key, float(losses[key]))
                if k%50==0:
                    writer.add_scalar(key, losses[key].item(), k)
            pbar.set_description(loss_info)

            if k % 100 == 0:
                grids = {}
                grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                grids['landmarks_gt'] = torchvision.utils.make_grid(
                    utils.tensor_vis_landmarks(images[visind], landmarks[visind], valid=landmarks_valid[visind]))
                grids['landmarks2d'] = torchvision.utils.make_grid(
                    utils.tensor_vis_landmarks(images[visind], landmarks2d[visind], valid=landmarks_valid[visind]))
                grids['landmarks3d'] = torchvision.utils.make_grid(
                    utils.tensor_vis_landmarks(images[visind], landmarks3d[visind], valid=landmarks_valid[visind]))

                grid = torch.cat(list(grids.values()), 1)
                writer.add_image('grid', normalize(grid).clamp(0, 1), k)
                # grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                # grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
                # cv2.imwrite('{}/{}/{}.jpg'.format(savefolder, model_name, k), grid_image)

        # non-rigid fitting of all the parameters with 68 face landmarks, photometric loss and regularization terms.
        pbar = tqdm(range(200, self.config.n_iters))
        for k in pbar:
            if k == self.config.reduce_reg_iter:
                self.config.w_pose_reg *= 0.1
                self.config.w_comp_reg *= 0.1
                self.config.w_shape_reg *= 0.1
                self.config.w_exp_reg *= 0.1

            losses = {}
            vertices, landmarks2d, landmarks3d, unposed_vertices = self.flame(shape_params=shape, expression_params=exp, pose_params=pose, eye_pose_params=eye_pose,
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

            losses['lmk'] = utils.l2_distance(landmarks2d[:, :, :2], gt_landmark[:, :, :2], conf=landmarks_valid) * self.config.w_lmk
            shape_reg_loss = (torch.sum(shape ** 2) / 2) * self.config.w_shape_reg
            exp_reg_loss = (torch.sum(exp ** 2) / 2) * self.config.w_exp_reg
            pose_reg_loss = (torch.sum(pose ** 2) / 2 + torch.sum(eye_pose ** 2) / 2 + torch.sum(neck_pose ** 2) / 2) * self.config.w_pose_reg
            losses['reg'] = shape_reg_loss + exp_reg_loss + pose_reg_loss
            losses['comp_reg'] = torch.tensor(0.0, device=self.device)
            if component_pca_type is not None:
                component_reg = 0
                for key in component_params.keys():
                    component_reg += torch.sum(component_params[key] ** 2) / 2
                component_reg_loss = component_reg * self.config.w_comp_reg
                losses['comp_reg'] += component_reg_loss
            mouth_loss = self.mouth_loss(gt_landmark[:, :, :2], landmarks2d[:, :, :2], landmarks_valid) * self.config.w_mouth
            eye_loss = self.eye_loss(gt_landmark[:, :, :2], landmarks2d[:, :, :2], landmarks_valid) * self.config.w_eye
            losses['mouth'] = mouth_loss
            losses['eye'] = eye_loss

            # res_shape = (unposed_vertices - self.flame.v_template).view(-1, 1)
            # losses['resreg'] = (res_shape - self.X_XtXinv_Xt.matmul(res_shape)).view(5023,3).norm(dim=1).mean()*1e7
            # losses['pairwise'] = self.pairwise_loss(unposed_vertices)*10

            meshes = pytorch3d.structures.Meshes(unposed_vertices, faces)
            losses['smooth'] = _3d_losses.mesh_laplacian_smoothing(meshes) * self.config.w_smooth

            ## render
            albedos = self.flametex(tex) / 255.
            ops = self.render(vertices, trans_vertices, albedos, lights)
            predicted_images = ops['images']
            losses['pho'] = (image_masks * (ops['images'] - images).abs()).mean() * self.config.w_photo
            vgg_feats_orig = self.vgg16((image_masks * images)*255/2, resize_images=True, return_lpips=True)
            vgg_feats_ren = self.vgg16((image_masks * ops['images'])*255/2, resize_images=True, return_lpips=True)
            losses['vgg'] = F.mse_loss(vgg_feats_orig, vgg_feats_ren) * self.config.w_vgg

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt.zero_grad()
            all_loss.backward()
            e_opt.step()

            loss_info = ''
            for key in losses.keys():
                loss_info = loss_info + '{}: {:.4e}, '.format(key, float(losses[key]))
                if k%50==0:
                    writer.add_scalar(key, losses[key].item(), k)
            pbar.set_description(loss_info)

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
                writer.add_image('grid', normalize(grid).clamp(0, 1), k)
                grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
                cv2.imwrite('{}/iterations/{}.jpg'.format(savefolder,  k), grid_image)

        occ_vt = self.render.image2verts(trans_vertices, occ_mask).detach()
        occ_verts_idx = occ_vt[:,:,0].bool()
        vis_verts_idx = ~ occ_verts_idx

        sampled_tex_vt = self.render.image2verts(trans_vertices, images).detach()
        sampled_tex_uv = self.render.world2uv(sampled_tex_vt)

        full_pose = torch.cat([pose[:, :3], neck_pose, pose[:, 3:], eye_pose], dim=1)
        fitted_params = {
            'shape': vertices.detach(),
            'unposed_vertices': unposed_vertices.detach(),
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
        mesh = Mesh(v=vertices[0].detach().cpu().numpy(), f=faces[0].cpu().numpy(), vc=1-occ_vt[0].expand(-1,3).detach().cpu().numpy())
        mesh.write_ply('{}/fitted.ply'.format(savefolder))
        save_image(normalize(sampled_tex_uv),
                   '{}/fitted_uv.jpg'.format(savefolder))

        # mesh =  Mesh(v=vertices[0].detach().cpu().numpy(), f=faces[0].cpu().numpy(), vt=self.vt, ft=self.ft,
        #              tex=normalize(sampled_tex_uv[0].cpu().numpy()))
        # mesh.write_ply('{}/fitted.obj'.format(savefolder))
        # save_ply('{}/fitted.ply'.format(savefolder), verts=vertices[0].detach().cpu(), faces=faces[0].cpu())
        # return single_params, shape_images
        # shape_error = (vertices[0] - shape_gt).norm() / np.sqrt(5023) * 1000
        # print('{} fitting shape error: {}'.format(model_name, shape_error))

        return fitted_params

    def run(self, imagepath, occmask=None, facemask=None, landmarkpath=None, shapegt=None):
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
        image = cv2.resize(cv2.imread(imagepath), (self.config.cropped_size, self.config.cropped_size)).astype(np.float32) / 255.
        image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
        images.append(torch.from_numpy(image[None, :, :, :]).to(self.device))
        images = torch.cat(images, dim=0)
        images = F.interpolate(images, [self.image_size, self.image_size])

        if occmask != None:
            occ_mask = cv2.resize(cv2.imread(occmask), (self.config.cropped_size, self.config.cropped_size)).astype(np.float32) / 255.
            occ_mask = occ_mask.transpose(2, 0, 1)[0:1]
            occ_masks.append(torch.from_numpy(occ_mask[None, :, :, :]).to(self.device))
            occ_masks = torch.cat(occ_masks, dim=0)
            occ_masks = threshold(F.interpolate(occ_masks, [self.image_size, self.image_size]))
            occ_dict = {'occ_mask': occ_masks}

        if facemask is not None:
            image_mask = np.load(facemask, allow_pickle=True)
            image_mask = image_mask[..., None].astype('float32')
            image_mask = image_mask.transpose(2, 0, 1)
            image_mask_bn = np.zeros_like(image_mask)
            image_mask_bn[np.where(image_mask != 0)] = 1.
            image_masks.append(torch.from_numpy(image_mask_bn[None, :, :, :]).to(self.device))
            image_masks = torch.cat(image_masks, dim=0)
            image_masks = F.interpolate(image_masks, [self.image_size, self.image_size])

        else:
            with torch.no_grad():
                face_parsing_out = self.unetseg(images*2-1)
                # face_mask = threshold(face_parsing_out[:,0:1])
                background_mask = threshold(face_parsing_out[:,2:3])
                image_masks = threshold(face_parsing_out[:,0:1])

                if occmask == None:
                    occ_masks = threshold(face_parsing_out[:,1:2])
                    occ_dict = {'occ_mask': occ_masks}

        if landmarkpath is not None:
            landmark = np.load(landmarkpath).astype(np.float32)
            landmarks.append(torch.from_numpy(landmark)[None, :, :].float().to(self.device))
            landmarks = torch.cat(landmarks, dim=0)
        else:
            with torch.no_grad():
                images_up = nn.functional.interpolate(images, size=(self.image_size, self.image_size)) * 2 - 1
                landmarks = self.fa.get_landmarks_from_batch(128 * (images_up + 1), detected_faces=self.detected_faces)
                landmarks = torch.tensor(np.stack([lmk[:68] for lmk in landmarks]), device=self.device)

        landmarks_valid = 1 - occ_masks[:,0,landmarks[0,:,1].long(),landmarks[0,:,0].long()]
        landmarks /= self.image_size
        landmarks = landmarks*2 - 1
        occ_dict['landmarks_valid'] = landmarks_valid
        occ_dict['landmarks_conf'] = landmarks_valid

        savefolder = os.path.sep.join([self.config.savefolder, '{}'.format(image_name)])
        utils.check_mkdir(savefolder)

        if shapegt is not None:
            shape_gt = Mesh(filename=shapegt)
            shape_gt.write_ply('{}/groundtruth.ply'.format(savefolder))
            shape_gt = torch.tensor(shape_gt.v).to(self.device)
        else:
            shape_gt=None

        # optimize
        print(savefolder, self.config.global_local_model)
        if self.config.global_local_model == 'full' or self.config.global_local_model == 'global':
            n_shape = self.config.shape_params
            n_exp = self.config.exp_params
            self.config.global_local_model = None
        else:
            n_shape = self.config.n_global_shape_params
            n_exp = self.config.n_global_exp_params

        fitted_params = self.fit_visible(images, landmarks, image_masks, occ_dict, savefolder, n_shape=n_shape, n_exp=n_exp,
                                         component_pca_type=self.config.global_local_model, shape_gt=shape_gt)
        np.save(savefile, fitted_params, allow_pickle=True)
        # self.diversify(fitted_params, images, landmarks, image_masks, occ_dict, savefolder, pca_config[2], shape_gt=shape_gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### input args
    parser.add_argument('--image', type=str, default=None, required=True)
    parser.add_argument('--occ-mask', type=str, default=None)
    parser.add_argument('--face-mask', type=str, default=None)
    parser.add_argument('--savefolder', type=str, default=None)
    parser.add_argument('--meshconv-config', type=str, default=None)
    parser.add_argument('--shape-gt', type=str, default=None)

    # model args
    parser.add_argument('--global-local-model', type=str,
                        default='component_se10_pca')
    parser.add_argument('--n-global-shape-params', type=int, default=10)
    parser.add_argument('--n-global-exp-params', type=int, default=10)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--cropped-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--flame-model-path', type=str, default='./data/generic_model.pkl')
    parser.add_argument('--flame-lmk-embedding-path', type=str, default='./data/landmark_embedding.npy')
    parser.add_argument('--tex-space-path', type=str, default='./data/FLAME_texture.npz')
    parser.add_argument('--camera-params', type=int, default=3)
    parser.add_argument('--shape-params', type=int, default=300)
    parser.add_argument('--exp-params', type=int, default=100)
    parser.add_argument('--pose-params', type=int, default=6)
    parser.add_argument('--tex-params', type=int, default=50)
    parser.add_argument('--use-face-contour', action='store_true')

    # training args
    parser.add_argument('--n-iters', type=int, default=1000)
    parser.add_argument('--w-photo', type=float, default=16)
    parser.add_argument('--w-vgg', type=float, default=1e7)
    parser.add_argument('--w-lmk', type=float, default=5)
    parser.add_argument('--w-mouth', type=float, default=1.0)
    parser.add_argument('--w-eye', type=float, default=1.0)
    parser.add_argument('--w-shape-reg', type=float, default=1e-2)
    parser.add_argument('--w-exp-reg', type=float, default=1e-2)
    parser.add_argument('--w-pose-reg', type=float, default=1e-2)
    parser.add_argument('--w-comp-reg', type=float, default=1e-2)
    parser.add_argument('--w-smooth', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default='0.005')
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--reduce-reg-iter', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    utils.check_mkdir(args.savefolder)
    fitting = PhotometricFitting(args)
    fitting.run(args.image, occmask=args.occ_mask, facemask=args.face_mask, shapegt=args.shape_gt)
