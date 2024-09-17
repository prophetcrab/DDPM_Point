from PIL import Image
from torch import nn
from GaussianDiffusion.Diffusion import *
from Utils.utils import *


class Diffusion(object):
    _defaults = {
        # -----------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        # -----------------------------------------------#
        "model_path": r'D:\PythonProject2\DDPM_Point\Logs\Model\diffusion_model_last_epoch_weights.pth',
        # -----------------------------------------------#
        #   输入的点云通道数
        # -----------------------------------------------#
        "channel":3,
        # -----------------------------------------------#
        #   输入点云大小的设置
        # -----------------------------------------------#
        "input_shape": 4096,
        # -----------------------------------------------#
        #   betas相关参数
        # -----------------------------------------------#
        "schedule": "cosine",
        "num_timesteps": 1000,
        "schedule_low": 1e-4,
        "schedule_high": 0.02,
        # -----------------------------------------------#
        #   AttentionModel参数
        # -----------------------------------------------#
        "n_layers": 6,
        "hidden_dim": 512,
        "num_heads": 4,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    # ---------------------------------------------------#
    #   初始化Diffusion
    # ---------------------------------------------------#

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        self.generate()

        show_config(**self._defaults)

    def generate(self):
        # ----------------------------------------#
        #   创建Diffusion模型
        # ----------------------------------------#
        if self.schedule == "cosine":
            betas = generate_cosine_schedule(self.num_timesteps)
        else:
            betas = generate_linear_schedule(
                self.num_timesteps,
                self.schedule_low * 1000 / self.num_timesteps,
                self.schedule_high * 1000 / self.num_timesteps,
            )

        self.net = GaussianDiffusion(
            AttentionModel(n_layers=self.n_layers, hidden_dim=self.hidden_dim, num_heads=self.num_heads),
            data_size=self.input_shape,
            data_channels=self.channel,
            betas=betas,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model loaded'.format(self.model_path))

        if self.cuda:
            self.net = self.net.cuda()

    def generate_sample_result(self, save_path, batch_size=1, data_size=1024, channels=3):
        with (torch.no_grad()):

            randn_in = torch.randn(batch_size, data_size, channels).cuda() if self.cuda else torch.randn(batch_size, data_size, channels)

            sample = self.net.sample(batch_size=1, device=device, use_ema=False)

            sample = sample[0].cpu().permute(1, 0).data.numpy()

            np.savetxt(save_path + "\SamplePoints", sample, delimiter=',' )

            visualize_point_cloud(sample)

            return sample

if __name__ == '__main__':
    Diffusion = Diffusion()
    Diffusion.generate()
    sample = Diffusion.generate_sample_result(save_path=r"D:\PythonProject2\DDPM_Point\Logs\Samples")
    print(sample.shape)
