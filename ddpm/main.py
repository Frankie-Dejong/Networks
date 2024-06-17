from torch.utils.data import DataLoader
from dataset import CelebAHQDataset, Cifar10Dataset, MnistDatasetWrapper
from unet import Unet
from diffusion import GaussianDiffusion, get_beta_schedule
from trainer import Trainer
import yaml
import argparse
import os
import torchvision


def get_dataloader(config):
    task = config["task"]
    batch_size = config["train"]["batch_size"]
    data_root = config["data"]["data_root"]
    if task == "celeba":
        train_set = CelebAHQDataset(data_root=data_root, train=True)
        test_set = CelebAHQDataset(data_root=data_root, train=False)
    elif task == "cifar-10":
        train_set = Cifar10Dataset(data_root=data_root, train=True)
        test_set = Cifar10Dataset(data_root=data_root, train=False)
    elif task == "mnist":
        train_set = MnistDatasetWrapper(data_root, train=True)
        test_set = MnistDatasetWrapper(data_root, train=False)
    else:
        raise NotImplementedError()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader


def get_model_and_diffusion(config):
    model = Unet(
        in_channels=config["model"]["in_channels"],
        out_channels=3 if config["task"] != "mnist" else 1,
        num_res_blocks=config["model"]["num_res_blocks"],
        attn_resolutions=config["model"]["attn_resolutions"],
        initial_resolutions=config["model"]["image_resolution"],
        dropout=config["model"]["dropout"],
        ch_mult=config["model"]["ch_mult"],
        resample_withconv=True
    )
    diffusion = GaussianDiffusion(
        get_beta_schedule(
            config["train"]["beta_schedule"], 
            config["train"]["beta_start"],
            config["train"]["beta_end"],
            config["train"]["num_diffusion_steps"],
        ),
        pred=config["train"]["pred"],
        loss_type=config["train"]["loss_type"]
    )
    return model, diffusion


def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict保存为yaml"""
    with open(save_path, 'w') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))

def main(config):
    train_loader, test_loader = get_dataloader(config)
    model, diffusion = get_model_and_diffusion(config)
    trainer = Trainer(
        exp_name=config["exp_name"], 
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=config["train"]["lr"],
        global_steps=config["train"]["global_step"],
        batch_size=config["train"]["batch_size"],
        log_step=config["train"]["log_step"],
        eval_step=config["train"]["eval_step"],
        save_step=config["train"]["save_step"],
        warm_up_step=config["train"]["warm_up_step"],
        resume_from= config["train"].get("resume_from", None),
        resume_step=config["train"].get("resume_step", 0)
    )
    
    save_dict_to_yaml(config, os.path.join(trainer.workdir, 'config_this_exp.yaml'))
    
    trainer.run()
    trainer.eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str)
    args = parser.parse_args()
    with open(args.config_name, mode='r') as f:
        config = yaml.safe_load(f)
    print("==========Config Start==========")
    print(config)
    print("==========Config End==========")
    main(config)