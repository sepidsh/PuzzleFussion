import random
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json
import os
import cv2 as cv
import csv
from tqdm import tqdm
from shapely import geometry as gm
from shapely.ops import unary_union
from collections import defaultdict
from glob import glob

def rotate_points(points, indices):
    indices = np.argmax(indices,1)
    indices[indices==0] = 1000
    unique_indices = np.unique(indices)
    num_unique_indices = len(unique_indices)
    rotated_points = np.zeros_like(points)
    rotation_angles = []
    for i in unique_indices:
        idx = (indices == i)
        selected_points = points[idx]
        rotation_angle = 0 if i==1 else (np.random.rand() * 360)
        # rotation_angle = 0 
        # rotation_angle = 0 if i==0 else (np.random.randint(4) * 90)
        rotation_angle = np.deg2rad(rotation_angle)
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)], # this is selected for return
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        rotated_selected_points = np.matmul(rotation_matrix, selected_points.T).T
        rotated_points[idx] = rotated_selected_points
        # rotation_matrix[0,1] = 1 if rotation_angle<np.pi else -1
        rotation_angles.extend(rotation_matrix[0:1].repeat(rotated_selected_points.shape[0], axis=0))
    return rotated_points, rotation_angles


def load_crosscut_data(
    batch_size,
    set_name,
    rotation 
):
    """
    For a dataset, create a generator over (shapes, kwargs) pairs.
    """
    print(f"loading {set_name} of crosscut...")
    deterministic = False if set_name=='train' else True
    dataset = CrosscutDataset(set_name, rotation=rotation)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
        )
    while True:
        yield from loader

class CrosscutDataset(Dataset):
    def __init__(self, set_name, rotation):
        super().__init__()
        max_num_points = 100
        base_dir = f'../datasets/cross_cut/{set_name}_data'
        self.set_name = set_name
        self.rotation = rotation
        self.puzzles = []
        self.rels = []
        lines_dir = glob(f'{base_dir}/*')
        for directory in lines_dir:
            puzzles = glob(f'{directory}/*')
            for puzzle_name in puzzles:
                # with open(f'{puzzle_name}/ground_truth_puzzle.csv') as csvfile:
                with open(f'{puzzle_name}/err1_s.csv') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    puzzle_dict = defaultdict(list)
                    puzzle = []
                    for row in reader:
                        if row[0] == 'piece':
                            continue
                        puzzle_dict[float(row[0])].append([float(row[1]),float(row[2])])
                    for piece in puzzle_dict.values():
                        piece = np.array(piece) / 100. - 0.5 # [[x0,y0],[x1,y1],...,[x15,y15]] and map to 0-1 - > -0.5, 0.5
                        piece = piece * 2 # map to [-1, 1]
                        center = np.mean(piece, 0)
                        piece = piece - center
                        puzzle.append({'poly': piece, 'center': center})
                    self.puzzles.append(puzzle)
                start_points = [0]
                for i in range(len(puzzle)-1):
                    start_points.append(start_points[-1]+len(puzzle[i]['poly']))
                with open(f'{puzzle_name}/ground_truth_rels.csv') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    rels = []
                    for row in reader:
                        if row[0] == 'piece1':
                            continue
                        [p1, e1, p2, e2] = [int(x) for x in row]
                        p11 = puzzle[p1]['poly'][e1]+puzzle[p1]['center']
                        p12 = puzzle[p1]['poly'][(e1+1)%len(puzzle[p1]['poly'])] + puzzle[p1]['center']
                        p21 = puzzle[p2]['poly'][e2]+puzzle[p2]['center']
                        p22 = puzzle[p2]['poly'][(e2+1)%len(puzzle[p2]['poly'])] + puzzle[p2]['center']
                        if np.abs(p11-p21).sum()<np.abs(p11-p22).sum():
                            rels.append([start_points[p1]+e1, start_points[p2]+e2])
                            rels.append([start_points[p1]+(e1+1)%(len(puzzle[p1]['poly'])), start_points[p2]+(e2+1)%(len(puzzle[p2]['poly']))])
                        else:
                            rels.append([start_points[p1]+e1, start_points[p2]+(e2+1)%(len(puzzle[p2]['poly']))])
                            rels.append([start_points[p1]+(e1+1)%(len(puzzle[p1]['poly'])), start_points[p2]+e2])
                    padding = np.zeros((100-len(rels), 2))
                    rels = np.concatenate((np.array(rels), padding), 0)
                    self.rels.append(rels)

        get_one_hot = lambda x, z: np.eye(z)[x]
        puzzles = []
        self_masks = []
        gen_masks = []
        for p in tqdm(self.puzzles):
            puzzle = []
            corner_bounds = []
            num_points = 0
            for i, piece in enumerate(p):
                poly = piece['poly']
                center = np.ones_like(poly) * piece['center']

                # Adding conditions
                num_piece_corners = len(poly)
                piece_index = np.repeat(np.array([get_one_hot(len(puzzle)+1, 32)]), num_piece_corners, 0)
                corner_index = np.array([get_one_hot(x, 32) for x in range(num_piece_corners)])

                # Adding rotation
                if self.rotation:
                    poly, angles = rotate_points(poly, piece_index)

                # Src_key_padding_mask
                padding_mask = np.repeat(1, num_piece_corners)
                padding_mask = np.expand_dims(padding_mask, 1)

                # Generating corner bounds for attention masks
                connections = np.array([[i,(i+1)%num_piece_corners] for i in range(num_piece_corners)])
                connections += num_points
                corner_bounds.append([num_points, num_points+num_piece_corners])
                num_points += num_piece_corners
                piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections), 1)
                puzzle.append(piece)
            
            puzzle_layouts = np.concatenate(puzzle, 0)
            if len(puzzle_layouts)>max_num_points:
                assert False
            padding = np.zeros((max_num_points-len(puzzle_layouts), 73))
            gen_mask = np.ones((max_num_points, max_num_points))
            gen_mask[:len(puzzle_layouts), :len(puzzle_layouts)] = 0
            puzzle_layouts = np.concatenate((puzzle_layouts, padding), 0)
            self_mask = np.ones((max_num_points, max_num_points))

            for i in range(len(corner_bounds)):
                self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[i][0]:corner_bounds[i][1]] = 0
            puzzles.append(puzzle_layouts)
            self_masks.append(self_mask)
            gen_masks.append(gen_mask)
        
        self.max_num_points = max_num_points
        self.puzzles = puzzles
        self.self_masks = self_masks
        self.gen_masks = gen_masks
        self.num_coords = 4

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        arr = self.puzzles[idx][:, :self.num_coords]
        polys = self.puzzles[idx][:, self.num_coords:self.num_coords+2]
        # if self.rotation:
        #     polys, angles = rotate_points(polys, self.puzzles[idx][:, self.num_coords+34:self.num_coords+66])
        #     arr = np.concatenate([arr, angles], 1)
        cond = {
                'self_mask': self.self_masks[idx],
                'gen_mask': self.gen_masks[idx],
                # 'poly': self.puzzles[idx][:, self.num_coords:self.num_coords+2],
                'poly': polys,
                'corner_indices': self.puzzles[idx][:, self.num_coords+2:self.num_coords+34],
                'room_indices': self.puzzles[idx][:, self.num_coords+34:self.num_coords+66],
                'src_key_padding_mask': 1-self.puzzles[idx][:, self.num_coords+66],
                'connections': self.puzzles[idx][:, self.num_coords+67:self.num_coords+69],
                'rels': self.rels[idx],
                }
        if self.set_name == 'train':
            pass
            #### Random Rotate
            # rotation = random.randint(0,3)
            # if rotation == 1:
            #     arr[:, [0, 1]] = arr[:, [1, 0]]
            #     arr[:, 0] = -arr[:, 0]
            # elif rotation == 2:
            #     arr[:, [0, 1]] = -arr[:, [1, 0]]
            # elif rotation == 3:
            #     arr[:, [0, 1]] = arr[:, [1, 0]]
            #     arr[:, 1] = -arr[:, 1]

            ## To generate any rotation uncomment this

            # if self.non_manhattan:
                # theta = random.random()*np.pi/2
                # rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                             # [np.sin(theta), np.cos(theta), 0]])
                # arr = np.matmul(arr,rot_mat)[:,:2]

            # Random Scale
            # arr = arr * np.random.normal(1., .5)

            # Random Shift
            # arr[:, 0] = arr[:, 0] + np.random.normal(0., .1)
            # arr[:, 1] = arr[:, 1] + np.random.normal(0., .1)

        arr = np.transpose(arr, [1, 0])
        return arr.astype(float), cond

if __name__ == '__main__':
    dataset = CrosscutDataset('test')
