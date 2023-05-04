import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.decomposition import PCA
from hyperparameter import z_dim



def visualize_z_latent_timestep(zs_latent : np.ndarray,model_name : str,  z_dim : int)->None:

    legend_label = ['left_task', 'right_task']
    if z_dim < 8:
        figuresize = [9, 10]
        row_num = z_dim
        column_num = 1
    elif z_dim > 8 and z_dim < 12:
        figuresize = [22, 14]
        row_num = 6
        column_num = 2
    elif z_dim  > 12 and z_dim < 20:
        figuresize = [22, 24]    
        row_num = int(z_dim/2)+1
        column_num = 2
    else:
        figuresize = [32, 36]    
        row_num = int(z_dim/2)+1
        column_num = 2
    fig = plt.figure(figsize=figuresize)
    plt.subplots_adjust(hspace=2, wspace=0.3)

    for i in range(z_dim):
        ax = fig.add_subplot(row_num, column_num, i+1)
        ax.plot(zs_latent[0, : , i])
        ax.plot(zs_latent[1, :, i])
        ax.set_xlabel("time_step")
        ax.set_title(f"dim{i+1}")
        ax.legend(legend_label)
        ax.legend(legend_label, loc='center left', bbox_to_anchor=(1., .5))
        
    plt.savefig("image/"+model_name +"/latent_timestep.png", bbox_inches='tight')


def visualize_randomly_task_latent(model, image_data : str,model_name : str, z_dim : int) -> None:
    image_data = image_data.reshape(image_data.shape[0]*image_data.shape[1], 3, 32, 32)
    model.eval()
    image_data = torch.from_numpy(image_data).float()
    image_data = image_data.to('cpu')
    with torch.no_grad():
        z_latent, _ = model.vision_vae.forward(image_data)#(16000, z_dim)

    z_latent = z_latent.reshape(-1, 80, z_dim)
    #joint_predictions = torch.zeros((n, 159, 14))
    random_index_left = np.random.randint(low=0, high=int(z_latent.shape[0]/2), size=(1,))
    random_index_right = np.random.randint(low=int(z_latent.shape[0]/2), high=int(z_latent.shape[0]), size=(1,))
    zs_latent_timestep = np.zeros((2, 80, z_dim))

    zs_latent_timestep[0] = z_latent[random_index_left]
    zs_latent_timestep[1] = z_latent[random_index_right]

    visualize_z_latent_timestep(zs_latent_timestep, model_name, z_dim)