from jittor.dataset import DataLoader
from dataset import CelebAHQDataset, Cifar10Dataset
from unet import Unet
from diffusion import GaussianDiffusion, get_beta_schedule
from trainer import Trainer
import yaml
import argparse
import os
import jittor as jit
jit.flags.use_cuda = 1


def get_model_and_diffusion(config):
    model = Unet(
        in_channels=config["model"]["in_channels"],
        out_channels=3,
        num_res_blocks=config["model"]["num_res_blocks"],
        attn_resolutions=config["model"]["attn_resolutions"],
        initial_resolutions=config["model"]["image_resolution"],
        dropout=config["model"]["dropout"],
        ch_mult=config["model"]["ch_mult"],
        resample_withconv=True
    )
    diffusion = GaussianDiffusion(get_beta_schedule(
        config["train"]["beta_schedule"], 
        config["train"]["beta_start"],
        config["train"]["beta_end"],
        config["train"]["num_diffusion_steps"]
        )
    )
    return model, diffusion


def main(config):
    model, diffusion = get_model_and_diffusion(config)
    trainer = Trainer(
        exp_name=config["exp_name"], 
        model=model,
        diffusion=diffusion,
        train_loader=None,
        test_loader=None,
        lr=config["train"]["lr"],
        global_steps=config["train"]["global_step"],
        batch_size=config["train"]["batch_size"],
        log_step=config["train"]["log_step"],
        eval_step=config["train"]["eval_step"],
        save_step=config["train"]["save_step"],
    )
    
    model.load_parameters(jit.load(config["model_path"]))
    trainer.generate(batch_size=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str)
    parser.add_argument('--exp_name', type=str)
    args = parser.parse_args()
    with open(args.config_name, mode='r') as f:
        config = yaml.safe_load(f)
    config['exp_name'] = args.exp_name
    print("==========Config Start==========")
    print(config)
    print("==========Config End==========")
    main(config)