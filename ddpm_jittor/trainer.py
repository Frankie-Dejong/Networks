import jittor as jit
from jittor import nn
from unet import Unet
from diffusion import GaussianDiffusion
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Trainer:
    def __init__(
        self,
        exp_name,
        model: Unet, 
        diffusion: GaussianDiffusion,
        train_loader: jit.dataset.DataLoader,
        test_loader: jit.dataset.DataLoader,
        lr=0.00002,
        global_steps=5e5,
        batch_size=64,
        log_step=1,
        eval_step=30,
        save_step=50
        ):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = nn.Adam(self.model.parameters(), lr=lr)
        self.global_steps = global_steps
        self.epochs = self.global_steps // batch_size
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epoch = 0
        self.log_step = log_step
        self.eval_step = eval_step
        self.save_step = save_step
        self.workdir = os.path.join('.', exp_name)
        self.res_dir = os.path.join('.', exp_name, 'results')
        self.ckpt_dir = os.path.join('.', exp_name, 'checkpoints')
        self.train_losses = []
        self.test_losses = []
        os.makedirs(self.workdir)
        os.makedirs(self.res_dir)
        os.makedirs(self.ckpt_dir)
        print(f"All the results will be saved in the dir {self.workdir}")
        
        
    def _denoise(self, x, t):
        B, H, W, C = x.shape
        assert t.shape == [B]
        out = self.model(x, t)
        assert out.shape == [B, H, W, C]
        return out
    
    def loss_fn(self, x):
        B, H, W, C = x.shape
        t = jit.randint(0, self.diffusion.num_timesteps, shape=[B])
        loss = self.diffusion.p_losses(
            denoise_fn=self._denoise, x_start=x, t=t
        )
        return loss
    
    def sample_fn(self, shape):
        return self.diffusion.p_sample_loop(self._denoise, shape)
    
    def run(self):
        print("==========Training start==========")
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.epoch = epoch
            for images in self.train_loader:
                loss = self.loss_fn(images)
                self.optimizer.step(loss)
            if self.epoch % self.log_step == 0:
                self.train_losses.append(loss.numpy()[0])
                tqdm.write(f'Loss: {loss.numpy()[0]}, Epoch: {self.epoch}')
            if self.epoch % self.eval_step == 0:
                self.eval()
                self.model.train()
            if self.epoch % self.save_step == 0:
                self.model.save(os.path.join(self.ckpt_dir, f'ckpt_at_epoch_{self.epoch}.pkl'))
            
        self.model.save(os.path.join(self.ckpt_dir, f'ckpt_at_epoch_{self.epoch}.pkl'))
        self.vis_losses()
        print("==========Training End==========")
            
                
    def eval(self):
        tqdm.write("==========Testing Start==========")
        self.model.eval()
        losses = []
        for images in self.test_loader:
            shape = images.shape
            losses.append(self.loss_fn(images).numpy()[0])
            
        # test_shape = shape
        # test_shape = [1, test_shape[1], test_shape[2], test_shape[3]]
        # generated_results = self.diffusion.p_sample_loop(self._denoise, test_shape)
        # save_dir = os.path.join(self.res_dir, f'epoch_{self.epoch}')
        # os.makedirs(save_dir)
        # for i, result in enumerate(generated_results):
        #     result = (result - (-1)) * (255 - 0) / (1 - (-1))
        #     result = np.array(result, dtype=np.int8)
        #     img = Image.fromarray(result.transpose(1, 2, 0))
        #     img.save(os.path.join(save_dir, f'image_{i}.png'))
        
        self.test_losses.append(sum(losses) / len(losses))
        tqdm.write(f"Loss on Val Set: {sum(losses) / len(losses)}, Epoch: {self.epoch}")
        tqdm.write("==========Testing End==========")
        
    def generate(self, batch_size):
        print(f"Generating {batch_size} images")
       
        test_shape = [batch_size, 3, 32, 32]
        result = self.diffusion.p_sample_loop(self._denoise, test_shape)
        result = (result - (-1)) * (255 - 0) / (1 - (-1))
        result = np.array(result, dtype=np.int8)
        for res in result:
            img = Image.fromarray(res.transpose(1, 2, 0))
            img.save(os.path.join(self.workdir, f'image_{i}.png'))
                
        
        
    def vis_losses(self):
        plt.plot(range(len(self.train_losses)), self.train_losses)
        plt.xlabel(f'epoch / {self.log_step}')
        plt.ylabel('train loss')
        plt.savefig(os.path.join(self.workdir, 'train_losses.png'))
        
        plt.clf()
        plt.plot(range(len(self.test_losses)), self.test_losses)
        plt.xlabel(f'epoch / {self.eval_step}')
        plt.ylabel('test loss')
        plt.savefig(os.path.join(self.workdir, 'test_losses.png'))
    
    