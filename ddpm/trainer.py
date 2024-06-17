import torch
from torch import nn
from unet import Unet
from diffusion import GaussianDiffusion
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.swa_utils import AveragedModel


class Trainer:
    def __init__(
        self,
        exp_name,
        model: Unet, 
        diffusion: GaussianDiffusion,
        train_loader,
        test_loader,
        lr=0.00002,
        global_steps=5e5,
        batch_size=64,
        log_step=1,
        eval_step=30,
        save_step=50,
        warm_up_step=10,
        resume_from=None,
        resume_step=0
        ):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0)
        self.initial_lr = lr
        self.lr_scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.global_steps = global_steps
        self.epochs = self.global_steps // batch_size
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epoch = 0
        self.log_step = log_step
        self.eval_step = eval_step
        self.save_step = save_step
        self.warm_up_step = warm_up_step
        self.workdir = os.path.join('.', exp_name)
        self.res_dir = os.path.join('.', exp_name, 'results')
        self.ckpt_dir = os.path.join('.', exp_name, 'checkpoints')
        self.log_dir = os.path.join('.', exp_name, 'logs')
        os.makedirs(self.workdir, exist_ok=True)
        os.makedirs(self.res_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.model = self.model.cuda()
        self.step = 0
        self.start_epoch = 0
        if resume_from is not None:
            print("Resume From {}".format(resume_from))
            self.step = resume_step
            self.start_epoch = resume_step // batch_size
            self.model.load_state_dict(torch.load(os.path.join(resume_from, 'ckpt.pth')))
            self.optimizer.load_state_dict(torch.load(os.path.join(resume_from, 'optimizer.pth')))
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(resume_from, 'scheduler.pth')))
        print(f"All the results will be saved in the dir {self.workdir}")
        
        
    def _denoise(self, x, t):
        B, H, W, C = x.shape
        assert t.shape == (B,)
        x = x.to(torch.float32)
        out = self.model(x, t)
        assert out.shape == x.shape
        return out
    
    def loss_fn(self, x):
        B, H, W, C = x.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,)).to(x.device)
        loss = self.diffusion.p_losses(
            denoise_fn=self._denoise, x_start=x, t=t
        )
        return loss * 15
    
    def sample_fn(self, shape):
        return self.diffusion.p_sample_loop(self._denoise, shape)
    
    def run(self):
        print("==========Training start==========")
        self.model.train()
        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            self.epoch = epoch
            if self.step < self.warm_up_step:
                for pgroup in self.optimizer.param_groups:
                    pgroup['lr'] = self.initial_lr * self.step / self.warm_up_step
            for images in self.train_loader:
                images = images.to('cuda')
                self.optimizer.zero_grad()
                loss = self.loss_fn(images)
                loss.backward()
                self.optimizer.step()
                if self.step % self.log_step == 0:
                    self.writer.add_scalar('train/loss', loss.cpu().item(), self.step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.step)
                self.step += 1

            if self.step > self.warm_up_step and self.optimizer.param_groups[0]['lr'] > 1e-5:
                self.lr_scheduler.step()
            if self.epoch % self.eval_step == 0:
                self.eval()
                self.model.train()
            if self.epoch % self.save_step == 0:
                os.makedirs(os.path.join(self.ckpt_dir, str(self.epoch)), exist_ok=True)
                save_dir = os.path.join(self.ckpt_dir, str(self.epoch))
                torch.save(self.model.state_dict(), os.path.join(save_dir, f'ckpt.pth'))
                torch.save(self.optimizer.state_dict(), os.path.join(save_dir, f'optimizer.pth'))
                torch.save(self.lr_scheduler.state_dict(), os.path.join(save_dir, f'scheduler.pth'))
            
        os.makedirs(os.path.join(self.ckpt_dir, str(self.epoch)), exist_ok=True)
        save_dir = os.path.join(self.ckpt_dir, str(self.epoch))
        torch.save(self.model.state_dict(), os.path.join(save_dir, f'ckpt.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, f'optimizer.pth'))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(save_dir, f'scheduler.pth'))
        print("==========Training End==========")
            
    @torch.no_grad()            
    def eval(self):
        tqdm.write("==========Testing Start==========")
        self.model.eval()
        losses = []
        for images in self.test_loader:
            images = images.to('cuda')
            shape = images.shape
            losses.append(self.loss_fn(images).cpu().item())
        
        B, C, H, W = shape
        test_shape = shape
        test_shape = [4, C, H, W]
        generated_results = self.diffusion.p_sample_loop_with_trajectory(self._denoise, test_shape, 200)
        
        final_res = generated_results[-1].cpu().permute(0, 2, 3, 1)
        final_res = ((final_res * 0.5) + 0.5) * 255
        save_dir = os.path.join(self.res_dir, f'epoch_{self.epoch}')
        os.makedirs(save_dir, exist_ok=True)
        for i, result in enumerate(final_res):
            result = np.array(result, dtype=np.uint8).squeeze()
            img = Image.fromarray(result)
            img.convert('RGB').save(os.path.join(save_dir, f'image_{i}.png'))
        
        self.writer.add_scalar('test/loss', sum(losses) / len(losses), self.step)
        diffusion_process = []
        for image_at_t in generated_results:
            diffusion_process.append(image_at_t[0].cpu())
        diffusion_process = np.stack(diffusion_process, axis=2).reshape(C, H, -1) * 0.5 + 0.5
        if C == 1:
            diffusion_process = np.repeat(diffusion_process, 3, axis=0)
        self.writer.add_image('test/sample', diffusion_process, self.step)
        
        tqdm.write(f"Loss on Val Set: {sum(losses) / len(losses)}, Epoch: {self.epoch}")
        tqdm.write("==========Testing End==========")
        
    @torch.no_grad()
    def generate(self, batch_size=32, channels=3, save_dir='./generation'):
        shape = (batch_size, channels, 32, 32)
        generated_results = self.diffusion.p_sample_loop_with_trajectory(self._denoise, shape, 200)
        
        final_res = generated_results[-1].cpu().permute(0, 2, 3, 1)
        final_res = ((final_res * 0.5) + 0.5) * 255
        os.makedirs(save_dir, exist_ok=True)
        for i, result in enumerate(final_res):
            result = np.array(result, dtype=np.uint8).squeeze()
            if channels == 1:
                result = np.repeat(result, 3, axis=-1)
            img = Image.fromarray(result)
            img.convert('RGB').save(os.path.join(save_dir, f'image_{i}.png'))