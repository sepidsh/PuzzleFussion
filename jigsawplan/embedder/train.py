import torch as th
from tqdm import tqdm
from model import get_model
from dataset import load_data
import os
import argparse
from torchvision.utils import save_image

def main():
    args = create_argparser()
    loss_fn = th.nn.MSELoss(reduction='none')
    # loss_fn = th.nn.MSELoss()
    print(args)
    model = get_model(use_gpu=args.use_gpu)
    model.to(args.device)
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_data = load_data(batch_size=args.batch_size, set_name='train', rotation=True)
    for step in tqdm(range(args.total_steps)):
        optimizer.zero_grad()
        batch = next(train_data).to(args.device).float()
        outputs = model(batch)
        num_pos = (batch>0).sum()
        # loss_weight = (((batch>0) * (batch.numel() - (2 * num_pos))/num_pos) + 1).detach()
        loss_weight = ((batch>0)).detach()
        loss = loss_fn(outputs, batch)
        loss = (loss * loss_weight).mean()
        # loss = (loss).mean()
        loss.backward()
        optimizer.step()
        if step % args.log_interval==0:
            print(f'step: {step} \t loss: {loss.item()}')
        if (step+1) % args.save_interval==0:
            print("_____________")
            scheduler.step()
            print(f'step: {step/args.save_interval} \t lr: {scheduler.get_last_lr()}')
            os.makedirs('ckpts', exist_ok=True)
            os.makedirs('ckpts/new_exp_128_losscolor', exist_ok=True)
            th.save(model.state_dict(), 'ckpts/new_exp_128_losscolor/model.pt')
            model.eval()
            with th.no_grad():
                total_loss = 0
                batch = next(train_data).to(args.device).float()
                outputs = model(batch)
            model.train()
            print(f'saving images...')
            save_image(outputs, f'ckpts/new_exp_128_losscolor/{step}_pred.png')
            save_image(batch, f'ckpts/new_exp_128_losscolor/{step}_gt.png')
            print("_____________")


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action='store_true', default=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--total_steps', default=500000, type=int)
    parser.add_argument('--log_interval', default=5000, type=int)
    parser.add_argument('--save_interval', default=5000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    args = parser.parse_args()
    args.device = 'cuda' if args.use_gpu else 'cpu'
    return args

if __name__ == "__main__":
    main()
