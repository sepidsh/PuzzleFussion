import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
import os
import cv2 as cv
import csv
from tqdm import tqdm
from shapely import geometry as gm
from collections import defaultdict
from glob import glob
from scipy import ndimage
from puzzle_fusion.embedder.model import get_model
import torchvision

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
        rotation_degree = 0 if i==1 else (np.random.rand() * 360)
        # rotation_angle = 0 
        # rotation_angle = 0 if i==0 else (np.random.randint(4) * 90)
        rotation_angle = np.deg2rad(rotation_degree)
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)], # this is selected for return
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        rotated_selected_points = np.matmul(rotation_matrix, selected_points.T).T
        rotated_points[idx] = rotated_selected_points
        # rotation_matrix[0,1] = 1 if rotation_angle<np.pi else -1
        rotation_angles.extend(rotation_matrix[0:1].repeat(rotated_selected_points.shape[0], axis=0))
    return rotated_points, rotation_angles, rotation_degree


def load_crosscut_data(
    batch_size,
    set_name,
    rotation,
    use_image_features,
):
    """
    For a dataset, create a generator over (shapes, kwargs) pairs.
    """
    print(f"loading {set_name} of crosscut...")
    deterministic = False if set_name=='train' else True
    dataset = CrosscutDataset(set_name, rotation=rotation, use_image_features=use_image_features)
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
    def __init__(self, set_name, rotation, use_image_features):
        super().__init__()
        duplicate = False
        self.use_image_features = use_image_features
        get_one_hot = lambda x, z: np.eye(z)[x]
        max_num_points = 100
        base_dir = f'../datasets/cross_cut/{set_name}_poly_data'
        img_base_dir = f'../datasets/poly_data'
        self.set_name = set_name
        self.rotation = rotation
        self.sample_files = []

        if self.use_image_features:
            device = "cuda" if th.cuda.is_available() else "cpu"
            model = get_model('../puzzle_fusion/embedder/ckpts/new_exp_128_losscolor/model.pt', use_gpu=True)
            model.eval()
            transform = torchvision.transforms.ToTensor()
        #import pdb ; pdb.set_trace()
        lines_dir = glob(f'{base_dir}/*')
        for directory in lines_dir:
            puzzles = glob(f'{directory}/*')
            for puzzle_name in tqdm(puzzles):

                if self.use_image_features:
                    image_puzzle_name = f"{img_base_dir}/_puzzle_name_{puzzle_name.split('/')[4]}_{puzzle_name.split('/')[5]}.npz"
                    if not os.path.isfile(image_puzzle_name):
                        continue
                    if os.path.isfile(f"../datasets/processed/{puzzle_name.split('/')[4]}_{puzzle_name.split('/')[5]}.npz"):
                        self.sample_files.append(f"../datasets/processed/{puzzle_name.split('/')[4]}_{puzzle_name.split('/')[5]}.npz")
                        continue
                if self.use_image_features:
                    try:
                        img = np.load(image_puzzle_name)
                    except Exception:
                        continue
                with open(f'{puzzle_name}/ground_truth_puzzle.csv') as csvfile:
                ##### For having different noises change the above line to one of these below ###
                # with open(f'{puzzle_name}/err3_n.csv') as csvfile: #### error3
                # with open(f'{puzzle_name}/err2_n.csv') as csvfile: #### error2
                # with open(f'{puzzle_name}/err1_n.csv') as csvfile: #### error2
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
                        if self.use_image_features:
                            puzzle.append({'poly': piece, 'center': center, 'img': img[str(len(puzzle))]})
                        else:
                            puzzle.append({'poly': piece, 'center': center})
                    if duplicate:
                        num_duplicates = np.random.randint(3)+1
                        for d_indx in range(num_duplicates):
                            duplicate_idx = np.random.randint(len(puzzle))
                            puzzle.append(puzzle[duplicate_idx])
                start_points = [0]
                for i in range(len(puzzle)-1):
                    start_points.append(start_points[-1]+len(puzzle[i]['poly']))
                with open(f'{puzzle_name}/ground_truth_rels.csv') as csvfile:
                ##### For having different noises change the above line to one of these below ###
                # with open(f'{puzzle_name}/err3_n.csv') as csvfile: #### error3
                # with open(f'{puzzle_name}/err2_n.csv') as csvfile: #### error2
                # with open(f'{puzzle_name}/err1_n.csv') as csvfile: #### error2
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

                p = puzzle
                puzzle_img = []
                puzzle = []
                corner_bounds = []
                num_points = 0
                for i, piece in enumerate(p):
                    poly = piece['poly']
                    center = np.ones_like(poly) * piece['center']
                    if self.use_image_features:
                        img = piece['img']

                    # Adding conditions
                    num_piece_corners = len(poly)
                    piece_index = np.repeat(np.array([get_one_hot(len(puzzle)+1, 32)]), num_piece_corners, 0)
                    corner_index = np.array([get_one_hot(x, 32) for x in range(num_piece_corners)])

                    # Adding rotation
                    if self.rotation:
                        poly, angles, degree = rotate_points(poly, piece_index)
                        if self.use_image_features:
                            img = ndimage.rotate(img, degree, reshape=False)
                           

                    # Adding images
                    if self.use_image_features:
                        puzzle_img.append(img)
                        inputs = transform(img).to(device).float()
                        image_features = model(inputs.unsqueeze(0), pred_image=False).reshape(1,-1)
                        image_features = image_features.expand(poly.shape[0], image_features.shape[1]).cpu().data.numpy()

                    # Src_key_padding_mask
                    padding_mask = np.repeat(1, num_piece_corners)
                    padding_mask = np.expand_dims(padding_mask, 1)


                    # Generating corner bounds for attention masks
                    connections = np.array([[i,(i+1)%num_piece_corners] for i in range(num_piece_corners)])
                    connections += num_points
                    corner_bounds.append([num_points, num_points+num_piece_corners])
                    num_points += num_piece_corners
                    if self.use_image_features:
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections, image_features), 1)
                    else:
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections), 1)
                    puzzle.append(piece)
                
                puzzle_layouts = np.concatenate(puzzle, 0)
                if len(puzzle_layouts)>max_num_points:
                    assert False
                if self.use_image_features:
                    padding = np.zeros((max_num_points-len(puzzle_layouts), 73+128))
                else:
                    padding = np.zeros((max_num_points-len(puzzle_layouts), 73))
                gen_mask = np.ones((max_num_points, max_num_points))
                gen_mask[:len(puzzle_layouts), :len(puzzle_layouts)] = 0
                puzzle_layouts = np.concatenate((puzzle_layouts, padding), 0)
                self_mask = np.ones((max_num_points, max_num_points))

                for i in range(len(corner_bounds)):
                    self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[i][0]:corner_bounds[i][1]] = 0
                sample_dict = {'puzzle': puzzle_layouts, 'self_mask': self_mask, 'gen_mask': gen_mask, 'rels': rels, 'images': puzzle_img}
                np.savez_compressed(f"../datasets/processed/{puzzle_name.split('/')[4]}_{puzzle_name.split('/')[5]}", **sample_dict) 
                self.sample_files.append(f"../datasets/processed/{puzzle_name.split('/')[4]}_{puzzle_name.split('/')[5]}.npz")
        self.num_coords = 4
        
        #self.sample_files = self.sample_files[:10000]
        self.samples = []
        for file in tqdm(self.sample_files, desc="loading processed dataset..."):
            sample = dict(np.load(file))
            sample.pop('images', None)
            self.samples.append(sample)
          
        
    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        # sample = np.load(self.sample_files[idx])
        sample = self.samples[idx]
        puzzle = sample['puzzle']
        # puzzle = self.samples[idx]
        arr = puzzle[:, :self.num_coords]
        polys = puzzle[:, self.num_coords:self.num_coords+2]
      
        cond = {
                'self_mask': sample['self_mask'],
                'gen_mask': sample['gen_mask'],
                # 'self_mask': self.self_masks[idx],
                # 'gen_mask': self.gen_masks[idx],
                'poly': polys,
                'corner_indices': puzzle[:, self.num_coords+2:self.num_coords+34],
                'room_indices': puzzle[:, self.num_coords+34:self.num_coords+66],
                'src_key_padding_mask': 1-puzzle[:, self.num_coords+66],
                'connections': puzzle[:, self.num_coords+67:self.num_coords+69],
                'rels': sample['rels'],
                # 'rels': self.rels[idx],
                }
        if self.use_image_features:
            cond['image_features'] = puzzle[:, -128:]
        arr = np.transpose(arr, [1, 0])
        return arr.astype(float), cond

if __name__ == '__main__':
    dataset = CrosscutDataset('train', rotation=True, use_image_features=True)
