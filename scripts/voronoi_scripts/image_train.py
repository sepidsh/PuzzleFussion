"""
Train a diffusion model on images.
"""

import argparse

import torch as th
from puzzle_fusion import logger, dist_util
from puzzle_fusion.resample import create_named_schedule_sampler
from puzzle_fusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    update_arg_parser,
)
from puzzle_fusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    update_arg_parser(args)

    dist_util.setup_dist()
    logger.configure(dir=f'ckpts/{args.exp_name}')

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.dataset=='crosscut':
        from puzzle_fusion.crosscut_dataset import load_crosscut_data
        data = load_crosscut_data(
            batch_size=args.batch_size,
            set_name=args.set_name,
            rotation=args.rotation,
            use_image_features=args.use_image_features,
        )
    elif args.dataset == 'voronoi':
        from puzzle_fusion.voronoi import load_voronoi_data
        data = load_voronoi_data(
            batch_size=args.batch_size,
            set_name=args.set_name,
            rotation=args.rotation,
        )
    else:
        print('dataset not exist!')
        assert False

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()
    dist_util.cleanup()


def create_argparser():
    defaults = dict(
        dataset = '',
        schedule_sampler= "uniform", #"loss-second-moment", "uniform",
        lr=1e-3,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=500,
        save_interval=25000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    parser = argparse.ArgumentParser()
    defaults.update(model_and_diffusion_defaults())
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
