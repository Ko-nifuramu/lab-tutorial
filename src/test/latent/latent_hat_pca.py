import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.data_processing.dataset_dataloader import create_goal_image_data


#data(num_of_data, time_step, 2) -> return latent(time_step, 2)
def cal_step_meanlatent(latent_data : np) -> torch:
    z_mean_latent = torch.zeros(latent_data.shape[1], 2)
    latent_data_tensor = torch.from_numpy(latent_data)

    for i in range(latent_data.shape[1]):
        zs_step_mean = torch.mean(latent_data_tensor[:, i, :], axis = 0).reshape([-1, 2])           
        z_mean_latent[i] = zs_step_mean

    return z_mean_latent


def cal_task_mean_next_latent(model, image_data : np, joint_data : np):
    goal_image_data = create_goal_image_data(image_data)
    goal_image_data = goal_image_data.reshape(-1, 3, 32, 32)
    image_data = image_data.reshape(-1, 3, 32, 32)
    joint_data = joint_data.reshape(-1,5)
    
    model.eval()
    image_data = torch.from_numpy(image_data).float()
    image_data = image_data.to('cpu')
    goal_image_data = torch.from_numpy(goal_image_data).float()
    goal_image_data = goal_image_data.to('cpu')
    joint_data = torch.from_numpy(joint_data).float()
    joint_data = joint_data.to('cpu')
    
    with torch.no_grad():
        z_i_t, _ = model.vision_vae.forward(image_data)#(60*160, 2)z_i, y = self.vision_vae.forward(i)
        z_i_g, _ = model.vision_vae.forward(goal_image_data)
        mean_hat, log_var_hat, j_out = model.rnn.forward(z_i_t,z_i_g, joint_data)
        z_latent_next = model.vision_vae.reparameterize(mean_hat, log_var_hat)
        print(z_latent_next.shape)

    z_latent_pca, explained_variance_ratio = pca(z_latent_next.reshape(z_latent_next.shape[0], -1), n_components=2)
    left_task_z_pca = z_latent_pca[:int(z_latent_pca.shape[0]/2), :]#(30*160, 2)
    right_task_z_pca = z_latent_pca[int(z_latent_pca.shape[0]/2) :, :]#(30*160, 2)
    left_task_z_pca = left_task_z_pca.reshape(-1, 80, 2)
    right_task_z_pca = right_task_z_pca.reshape(-1, 80, 2)
    print("z_left_latent_pca")

    left_task_mean_latent = cal_step_meanlatent(left_task_z_pca)
    right_task_mean_latent = cal_step_meanlatent(right_task_z_pca)
    
    return left_task_mean_latent, right_task_mean_latent, explained_variance_ratio



def pca(z, n_components):
    pca = PCA(n_components).fit(z)
    return pca.fit_transform(z), pca.explained_variance_ratio_


#散布図にカラーマップ、ラベルをつけたりする
# https://python-academia.com/matplotlib-scatter/
#scatterとplotの二つを同時に適用できる
def visualize_nextlatent_2task_pca(left_pca,right_pca,explained_variance_ratio,  model_name):
    labels = [str(10*num) for num in range(1,int(left_pca.shape[0]/10)+1)]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(left_pca[:, 0], left_pca[:, 1])
    ax.scatter(right_pca[:, 0], right_pca[:, 1])
    ax.plot(left_pca[:, 0], left_pca[:, 1])
    ax.plot(right_pca[:, 0], right_pca[:, 1])
    ax.set_xlabel("first_pricipal_component(ratio = {})".format(explained_variance_ratio[0]))
    ax.set_ylabel("second_pricipal_component(ratio = {})".format(explained_variance_ratio[1]))
    ax.legend(['left_task', 'right_task'])
    ax.set_title('pca')

    for i, label in enumerate(labels):
        ax.text(left_pca[10*i, 0], left_pca[10*i, 1],label)
        ax.text(right_pca[10*i, 0], right_pca[10*i, 1],label)
    plt.savefig("image/"+model_name + "/z_latent_hat_2dim_pca")