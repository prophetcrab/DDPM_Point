import os.path

from GaussianDiffusion.DDPM import *

if __name__ == '__main__':
    save_path = r"D:\PythonProject2\DDPM_Point\Logs\Samples"

    sample_name = "mysample.txt"
    ddpm = Diffusion()

    sample = ddpm.generate_sample_result(os.path.join(save_path, sample_name))

    print(sample)

    print("Sample Done")