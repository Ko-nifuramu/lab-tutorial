import torch.nn as nn
import torch

#data_processing
trainsample_ratio = 0.8
joint_data_noise_std = 0.1
image_data_noise_std = 0.3

#model
z_dim = 48
joint_dim = 5
hidden_size = 64
num_layer = 1
vae_weight = 1e3
joint_weight = 5e6
kld_weight = 1e-1#1e-3
kld_hat_weight = 10#1e-1
model_name = "GRU_hid64_01_03_z"
is_thismodel_directory_exist = False
_activation = nn.Mish()

#train
batch_size = 10
lr = 0.0005
optimizer = torch.optim.Adam
epochs = 10000
early_stop_epoch = 3000
#gradient_clipping = 100
