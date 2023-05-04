import numpy as np
from torch.utils.data import dataloader
import torch
from src.data_processing.data_preprocessing import*
from torch.utils.data import DataLoader
from hyperparameter import joint_data_noise_std, trainsample_ratio, image_data_noise_std

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, vision_input_data :np,
                vision_target_data,
                joint_input_data : np, 
                joint_target_data : np,):
        
        self.v_input_data = torch.from_numpy(vision_input_data).float()
        self.v_target_data = torch.from_numpy(vision_target_data).float()
        self.j_input_data = torch.from_numpy(joint_input_data).float()
        self.j_target_data = torch.from_numpy(joint_target_data).float()
        self.v_goal_data = torch.from_numpy(create_goal_image_data(vision_target_data)).float()

    def __len__(self):
        return self.v_input_data.shape[0]

    def __getitem__(self, index):
        #ノイズを加える
        vision_input_datum = self.v_input_data[index] + torch.normal(mean=0, std=image_data_noise_std, size=self.v_input_data[index].shape)
        vision_target_datum = self.v_target_data[index]+ torch.normal(mean=0, std=image_data_noise_std, size=self.v_target_data[index].shape)
        joint_input_datum = self.j_input_data[index] + torch.normal(mean=0, std=joint_data_noise_std, size=self.j_input_data[index].shape)
        joint_target_datum = self.j_target_data[index]+ torch.normal(mean=0, std=joint_data_noise_std, size=self.j_target_data[index].shape)
        
        vision_goal_datum = self.v_goal_data[index]+ torch.normal(mean=0, std=joint_data_noise_std, size=self.v_goal_data[index].shape)

        #元の地域に戻す
        vision_input_datum = torch.clamp(vision_input_datum, min=0, max=1)
        joint_input_datum = torch.clamp(joint_input_datum, min=-0.95, max=0.95)#-1, 1付近も含めてしまうと活性化関数tanhで勾配消失が起きてしまう
        joint_target_datum = torch.clamp(joint_target_datum, min=-0.95, max = 0.95)

        return vision_input_datum, vision_target_datum, joint_input_datum, joint_target_datum, vision_goal_datum


def create_goal_image_data(vision_target_data : np)->np:
    if len(vision_target_data.shape) == 4:
        vision_goal_data = vision_target_data[ -1:, :, :, :]
    else:
        vision_goal_data_1step = vision_target_data[: , -1:, :, :, :]
    # print(f'vision_target_data : {vision_target_data.shape}')
        for i in range(vision_target_data.shape[1]):
            if i == 0:
                vision_goal_data =vision_goal_data_1step
                continue
            vision_goal_data = np.concatenate([vision_goal_data, vision_goal_data_1step], axis=1)
            
    return vision_goal_data

def create_dataloader() -> DataLoader:
    def g(image_data : np, joint_data : np, batch_size : int, dataset = MyDataset):
        joint_data, joint_eachdim_maxmin_array = joint_regularization(joint_data)
        ratio = trainsample_ratio      

        indeces = [int(image_data.shape[0] * n) for n in [ratio/2, 0.5, (1+ratio)/2]]
        train_image_data1, val_image_data1, train_image_data2, val_image_data2 = np.split(image_data[:, 0:-1, :, :, :], indeces, axis=0)
        train_image_data = np.concatenate([train_image_data1, train_image_data2], axis=0)
        val_image_data = np.concatenate([val_image_data1, val_image_data2], axis=0)
        
        train_image_data1, val_image_data1, train_image_data2, val_image_data2 = np.split(image_data[:, 1:, :, :, :], indeces, axis=0)
        train_image_target_data = np.concatenate([train_image_data1, train_image_data2], axis=0)
        val_image_target_data = np.concatenate([val_image_data1, val_image_data2], axis=0)
        
        train_joint_input1, val_joint_input1, train_joint_input2, val_joint_input2 = np.split(joint_data[:, 0:-1, :], indeces, axis=0)
        train_joint_input_data = np.concatenate([train_joint_input1, train_joint_input2], axis=0)
        val_joint_input_data = np.concatenate([val_joint_input1, val_joint_input2], axis=0)
        
        train_joint_input1, val_joint_input1, train_joint_input2, val_joint_input2 = np.split(joint_data[:, 1:, :], indeces, axis=0)
        train_joint_target_data = np.concatenate([train_joint_input1, train_joint_input2], axis=0)
        val_joint_target_data = np.concatenate([val_joint_input1, val_joint_input2], axis=0)
        
        train_dataset = MyDataset(train_image_data, train_image_target_data, train_joint_input_data, train_joint_target_data)
        val_dataset = MyDataset(val_image_data, val_image_target_data,val_joint_input_data, val_joint_target_data)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
        return train_dataloader, val_dataloader

    return g


