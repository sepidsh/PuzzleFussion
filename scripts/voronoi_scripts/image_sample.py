"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import argparse
import os

import numpy as np
import torch as th
import cv2

import io
import PIL.Image as Image
import drawsvg as drawsvg
import cairosvg
import imageio
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths


from puzzle_fusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    update_arg_parser,
)
from puzzle_fusion import dist_util
import webcolors
import networkx as nx
from collections import defaultdict
from shapely.geometry import Point, LineString
from shapely.geometry.base import geom_factory
from shapely.geos import lgeos

import random
th.manual_seed(0)
random.seed(0)
np.random.seed(0)

def rotate_points(points, cos_theta, sin_theta):
    shape = points.shape
    theta = -th.atan2(-sin_theta, cos_theta)
    cos_theta = th.cos(theta)
    sin_theta = -th.sin(theta)
    # theta = th.acos(cos_theta)
    # sin_theta[sin_theta>0] = 1
    # sin_theta[sin_theta<0] = -1
    # theta = theta * sin_theta
    # theta = -theta
    sin_theta = th.sin(theta)
    cos_theta = th.cos(theta)

    rotation_matrix = th.stack([
        th.stack([cos_theta, -sin_theta]),
        th.stack([sin_theta, cos_theta])
    ])
    rotation_matrix = rotation_matrix.permute([2,3,4,0,1])
    points = points.reshape(-1, 2, 1)
    rotation_matrix = rotation_matrix.reshape(-1, 2, 2)
    rotated_points = th.bmm(rotation_matrix.double(), points.double())
    return rotated_points.reshape(shape)

def save_samples(sample, ext, model_kwargs, rotation, tmp_count, save_gif=False, save_edges=False, ID_COLOR=None, save_svg=False):
    if not save_gif:
        sample = sample[-1:]
    for k in range(sample.shape[0]):
        if rotation:
            rot_s_total=[]
            rot_c_total=[]
            for nb in range(model_kwargs[f'room_indices'].shape[0]):
                array_a = np.array(model_kwargs[f'room_indices'][nb].cpu())
                room_types = np.where(array_a == array_a.max())[1]
                room_types = np.append(room_types, -10)
                rot_s =[]
                rot_c =[]
                rt =0
                no=0
                for ri in range(len(room_types)):
                    if rt!=room_types[ri]:
                        for nn in range(no):
                            rot_s.append(np.array(rot_s_tmp).mean())
                            rot_c.append(np.array(rot_c_tmp).mean())
                        rt=room_types[ri]
                        no=1
                        rot_s_tmp = [sample[k:k+1,:,:,3][0][nb][ri].cpu().data.numpy()]
                        rot_c_tmp = [sample[k:k+1,:,:,2][0][nb][ri].cpu().data.numpy()]
                    else:
                        no+=1
                        rot_s_tmp.append(sample[k:k+1,:,:,3][0][nb][ri].cpu().data.numpy())
                        rot_c_tmp.append(sample[k:k+1,:,:,2][0][nb][ri].cpu().data.numpy())
                while len(rot_s)<100:
                    rot_s.append(0)
                    rot_c.append(0)
                rot_s_total.append(rot_s)
                rot_c_total.append(rot_c)
            poly = rotate_points(model_kwargs['poly'].unsqueeze(0),th.unsqueeze(th.Tensor(rot_c_total).cuda(),0), th.unsqueeze(th.Tensor(rot_s_total).cuda(),0))
            # poly = rotate_points(model_kwargs['poly'].unsqueeze(0), sample[k:k+1,:,:,2], sample[k:k+1,:,:,3])
        else:
            poly = model_kwargs['poly'].unsqueeze(0)


        center_total = []
        for nb in range(model_kwargs[f'room_indices'].shape[0]):
            array_a = np.array(model_kwargs[f'room_indices'][nb].cpu())
            room_types = np.where(array_a == array_a.max())[1]
            room_types = np.append(room_types, -10)
            center =[]
            rt =0
            no=0
            for ri in range(len(room_types)):
                if rt!=room_types[ri]:
                    for nn in range(no):
                        center.append(np.array(center_tmp).mean(0))
                    rt=room_types[ri]
                    no=1
                    center_tmp = [sample[k:k+1,:,:,:2][0][nb][ri].cpu().data.numpy()]
                else:
                    no+=1
                    center_tmp.append(sample[k:k+1,:,:,:2][0][nb][ri].cpu().data.numpy())
            while len(center)<100:
                center.append([0, 0])
            center_total.append(center)

        sample[k:k+1,:,:,:2] = th.Tensor(center_total).cuda() + poly
        # sample[k:k+1,:,:,:2] = sample[k:k+1,:,:,:2] + poly
    sample = sample[:,:,:,:2]
    draw_ =False
    if draw_ == True:
        for i in tqdm(range(sample.shape[1])):
            resolution = 256
            images = []
            images2 = []
            images3 = []
            for k in range(sample.shape[0]):
                draw = drawsvg.Drawing(resolution, resolution, displayInline=False)
                draw.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
                draw2 = drawsvg.Drawing(resolution, resolution, displayInline=False)
                draw2.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
                draw3 = drawsvg.Drawing(resolution, resolution, displayInline=False)
                draw3.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
                draw_color = drawsvg.Drawing(resolution, resolution, displayInline=False)
                draw_color.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='white'))
                polys = []
                types = []
                for j, point in (enumerate(sample[k][i])):
                    if model_kwargs[f'src_key_padding_mask'][i][j]==1:
                        continue
                    point = point.cpu().data.numpy()
                    if j==0:
                        poly = []
                    if j>0 and (model_kwargs[f'room_indices'][i, j]!=model_kwargs[f'room_indices'][i, j-1]).any():
                        c = (len(polys)%28) + 1
                        polys.append(poly)
                        types.append(c)
                        poly = []
                    pred_center = False
                    if pred_center:
                        point = point/2 + 1
                        point = point * resolution//2
                    else:
                        point = point/2 + 0.5
                        point = point * resolution
                    poly.append((point[0], point[1]))
                c = (len(polys)%28) + 1
                polys.append(poly)
                types.append(c)
                for poly, c in zip(polys, types):
                    room_type = c
                    c = webcolors.hex_to_rgb(ID_COLOR[c])
                    draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='black', stroke_width=1))
                    draw.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill='black', fill_opacity=0.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                    draw2.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                    for corner in poly:
                        draw.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                        draw3.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                #images.append(Image.open(io.BytesIO(cairosvg.svg2png(draw.asSvg()))))
                #images2.append(Image.open(io.BytesIO(cairosvg.svg2png(draw2.asSvg()))))
                #images3.append(Image.open(io.BytesIO(cairosvg.svg2png(draw3.asSvg()))))
                if k==sample.shape[0]-1 or True:
                    if save_edges:
                        draw.save_svg(f'outputs/{ext}/{tmp_count+i}_{k}_{ext}.svg')
                    if save_svg:
                        draw_color.save_svg(f'outputs/{ext}/{tmp_count+i}c_{k}_{ext}.svg')
                    else:
                        Image.open(io.BytesIO(cairosvg.svg2png(draw_color.asSvg()))).save(f'outputs/{ext}/{tmp_count+i}c_{ext}.png')
            # if save_gif:
            #     imageio.mimwrite(f'outputs/gif/{tmp_count+i}.gif', images, fps=10, loop=1)
            #     imageio.mimwrite(f'outputs/gif/{tmp_count+i}_v2.gif', images2, fps=10, loop=1)
            #     imageio.mimwrite(f'outputs/gif/{tmp_count+i}_v3.gif', images3, fps=10, loop=1)
    return sample[-1]

def main():
    args = create_argparser().parse_args()
    update_arg_parser(args)
    dist_util.setup_dist()
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(th.device('cuda'))
    model.eval()

    errors = []
    iou_ = 0
    recal = 0
    pres  = 0
    items = 0

    for _ in range(1):
        print("sampling...")
        tmp_count = 0
        os.makedirs('outputs/pred', exist_ok=True)
        os.makedirs('outputs/gt', exist_ok=True)
        os.makedirs('outputs/gif', exist_ok=True)

        if args.dataset=='crosscut':
            from puzzle_fusion.crosscut_dataset import load_crosscut_data
            ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
                        6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 9: '#1F849B', 10: '#727171',
                        11: '#785A67', 12:'#D3A2C7', 13: '#ff55a3',14 : '#d7e8fc', 15: '#ff91af' ,
                        16 :'#d71868', 17: '#d19fe8', 18: '#00cc99', 19: '#eec8c8', 20:'#739373'}
           
            data = load_crosscut_data(
                batch_size=args.batch_size,
                set_name=args.set_name,
                rotation=args.rotation,
                use_image_features=args.use_image_features,
            )
        elif  args.dataset=='voronoi':
            from puzzle_fusion.voronoi import load_voronoi_data
        
            ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
                        6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 9: '#1F849B', 10: '#727171',
                        11: '#785A67', 12:'#D3A2C7', 13: '#ff55a3',14 : '#d7e8fc', 15: '#ff91af' ,
                        16 :'#d71868', 17: '#d19fe8', 18: '#00cc99', 19: '#eec8c8', 20:'#739373'}
           
            data = load_voronoi_data(
                batch_size=args.batch_size,
                set_name=args.set_name,
                rotation=args.rotation,
                
            )
        else:
            print("dataset does not exist!")
            assert False
        graph_errors = []
        while tmp_count < args.num_samples:
            model_kwargs = {}
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            data_sample, model_kwargs = next(data)
            for key in model_kwargs:
                model_kwargs[key] = model_kwargs[key].cuda()

            sample = sample_fn(
                model,
                data_sample.shape,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample_gt = data_sample.cuda().unsqueeze(0)
            sample = sample.permute([0, 1, 3, 2])
            sample_gt = sample_gt.permute([0, 1, 3, 2])
            gt = save_samples(sample_gt, 'gt', model_kwargs, args.rotation, tmp_count, ID_COLOR=ID_COLOR, save_svg=args.save_svg)
            pred = save_samples(sample, 'pred', model_kwargs, args.rotation, tmp_count, ID_COLOR=ID_COLOR, save_svg=args.save_svg)
            outs =(get_metric(gt, pred, model_kwargs))
            #import pdb ; pdb.set_trace()
            pres += outs[1][0]
            recal += outs[1][1]
            iou_ += outs[0]
            items +=1

            tmp_count+=sample_gt.shape[1]
        print( "overlap is:" , iou_/items)
        print( "precision is:" , pres/items)
        print( "recall is:" , recal/items)
        print("sampling complete")


def weighted_points(polys):
    areas = [np.full((len(poly), 1),
                     cv2.contourArea(poly.astype(np.float32)))
             for poly in polys]
    weights = np.vstack(areas)
    points = np.vstack(polys)
    return (weights, points)

def translate_to_gt(pieces_gt, pieces_sol):
    W, p_gt = weighted_points(pieces_gt)
    p_sol = np.vstack(pieces_sol)
    center_gt, center_sol = [np.sum(W * p, axis=0) / np.sum(W)
                             for p in [p_gt, p_sol]]
    X = p_sol - center_sol
    Y = p_gt - center_gt
    S = X.T @ np.diag(W.squeeze()) @ Y
    U, _, V = np.linalg.svd(S)
    R = (V @ np.array([[1, 0],
                       [0, np.linalg.det(V @ U.T)]]) @ U.T)
    t = center_gt - center_sol @ R.T
    return (R.T, t)

def piece_weights(pieces):
    areas = [cv2.contourArea(p.astype(np.float32)) for p in pieces]
    total = sum(areas)
    return [x / total for x in areas]

def overlap_score(pieces_gt, sol_transformed, weights):
    total_score = 0
    for p_gt, p_sol, w in zip(pieces_gt, sol_transformed, weights):
        overlap, _ = cv2.intersectConvexConvex(p_gt.astype(np.float32),
                                       p_sol.astype(np.float32))
        
        curr_area = cv2.contourArea(p_sol.astype(np.float32))
        if(curr_area > 0):
            score = overlap / curr_area
            total_score += score * w
    return total_score


def connection_score(gt, pred):
    assert len(gt)==len(pred)
    connections_gt = []
    connections_pred = []
    for i in range(len(gt)):
        for j in range(i+1, len(gt)):
            for k in range(gt[i].shape[0]):
                for l in range(gt[j].shape[0]):
                    p1 = LineString([gt[i][k], gt[i][(k+1)%gt[i].shape[0]]])
                    p2 = LineString([gt[j][l], gt[j][(l+1)%gt[j].shape[0]]])
                    if p1.hausdorff_distance(p2) < 0.1:
                        connections_gt.append([i, k, j, l])
                    p1 = LineString([pred[i][k], pred[i][(k+1)%pred[i].shape[0]]])
                    p2 = LineString([pred[j][l], pred[j][(l+1)%pred[j].shape[0]]])
                    if p1.hausdorff_distance(p2) < 0.1:
                        connections_pred.append([i, k, j, l])
    precision = np.sum([connections_pred[x] in connections_gt for x in range(len(connections_pred))])/max(len(connections_pred), 1)
    recall = np.sum([connections_pred[x] in connections_gt for x in range(len(connections_pred))])/max(len(connections_gt), 1)
    return np.array([precision, recall])

def get_metric(gt, pred, model_kwargs):
    gt_puzzles = []
    pred_puzzles = []
    overlap_score_total = 0
    connection_score_total = np.array([0,0])
    for i in range(gt.shape[0]):
        gt_puzzle = []
        pred_puzzle = []
        gt_poly = []
        pred_poly = []
        for j in range(gt.shape[1]):
            if j>0:
                if(model_kwargs['room_indices'][i][j].argmax()!=model_kwargs['room_indices'][i][j-1].argmax()) or j==gt.shape[1]-1:
                    gt_puzzle.append(np.array(gt_poly))
                    pred_puzzle.append(np.array(pred_poly))
                    gt_poly, pred_poly = [], []
            gt_poly.append([gt[i][j][0].cpu().data, gt[i][j][1].cpu().data])
            pred_poly.append([pred[i][j][0].cpu().data, pred[i][j][1].cpu().data])
        gt_puzzles.append(gt_puzzle[:-1]) # -1 for padding
        pred_puzzles.append(pred_puzzle[:-1]) # -1 for padding

        R, t = translate_to_gt(gt_puzzles[-1], pred_puzzles[-1]) # selecting the last one
        sol_transformed = [p @ R + t for p in pred_puzzles[-1]]
        connection_score_index = connection_score(gt_puzzles[-1], sol_transformed)
        connection_score_total = connection_score_total + connection_score_index
        overlap_score_index = overlap_score(gt_puzzles[-1], sol_transformed, piece_weights(gt_puzzles[-1]))
        overlap_score_total += overlap_score_index
        # print(i, overlap_score_index, connection_score_index)
    return overlap_score_total/gt.shape[0], connection_score_total/gt.shape[0]


def create_argparser():
    defaults = dict(
        dataset='',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        draw_graph=True,
        save_svg=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
