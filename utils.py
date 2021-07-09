import numpy as np
import imageio

def save_gif(env, file_name, fps=30):
    imgs = np.array(env.get_attr('images')[0]) # Images attribute packaged as list
    num_frames = np.array(env.get_attr('images')[0]).shape[0] # Shape is (frames, channels, x, y)
    list_imgs = [np.moveaxis(imgs[i,...],0,-1) for i in range(num_frames)] # Need list of (x, y, channels)
    imageio.mimsave(f'./videos/{file_name}.gif', list_imgs, fps=30)