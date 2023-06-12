import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    google_colab = False
    if google_colab:
      root = '/content/drive/MyDrive/nerf_desktop/Hololens2-NeRF/'
    else:
      root = './'
    
    error_path = os.path.join('./','model_error',f'error{num}.txt')
    
    errors = []
    with open(error_path,'r') as f:
        
        lines = f.readlines()
        for line in lines:
            error = float(line)
            errors.append(error)
    plt.plot(np.arange(len(errors)), errors)
    plt.show()
