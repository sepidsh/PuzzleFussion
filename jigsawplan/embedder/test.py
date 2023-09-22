
import torch as th
from tqdm import tqdm
from model import get_model
from dataset import load_data
import os
import argparse
from torchvision.utils import save_image

def main():
    args = create_argparser()
    print(args)
    model = get_model('ckpts/new_exp/model.pt', use_gpu=args.use_gpu)
    model.to(args.device)
    data = load_data(batch_size=args.batch_size, set_name='train', rotation=True)
    os.makedirs('output_images', exist_ok=True)
    model.eval()
    for step in range(10):
        with th.no_grad():
            batch = next(data).to(args.device).float()
            outputs = model(batch)
        save_image(outputs, f'output_images/{step}_pred.png')
        save_image(batch, f'output_images/{step}_gt.png')


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action='store_true', default=True)
    parser.add_argument('--batch_size', default=128, type=int)
    args = parser.parse_args()
    args.device = 'cuda' if args.use_gpu else 'cpu'
    return args

if __name__ == "__main__":
    main()
