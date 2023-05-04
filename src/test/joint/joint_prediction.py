import torch
import matplotlib.pyplot as plt
import numpy as np
from src.data_processing.data_preprocessing import joint_regularization, reverse_joint_regurlarization
import torch.nn as nn
from src.path import mkdir_ofimage_from_modelName
from hyperparameter import is_thismodel_directory_exist, joint_dim
from src.data_processing.dataset_dataloader import create_goal_image_data



#Simple RNN, GRU -> image:open, joint : closed
def test_predict_joint_closed_RNN(model, image_data : torch.Tensor, joint_data : torch.Tensor, joint_image_path : str, data_name : str, random_index_array : list):

    image_data_tensor = torch.from_numpy(image_data).clone()
    joint_data_tensor = torch.from_numpy(joint_data).clone()

    #TO DO : 右左一個ずつ生成したほうがいい
    data_num_in_1image = 1
    joint_predictions = torch.zeros((data_num_in_1image, 79, joint_dim))
    inputs_image = torch.zeros((data_num_in_1image, 80, 3, 32, 32))
    inputs_joint = torch.zeros((data_num_in_1image, 80, joint_dim))

    for i in range(data_num_in_1image):
        input_image = image_data_tensor[random_index_array[i]]
        input_joint = joint_data_tensor[random_index_array[i]]

        inputs_image[i] = input_image 
        inputs_joint[i] = input_joint

    joint_input_datum = inputs_joint[:, 0, :]
    joint_target_data = inputs_joint[:, 1 :, :]
    image_input_data = inputs_image[:, 0:-1, :, :, :]
    image_goal_datum = inputs_image[:, -1, :, :, :]#(data_num_in_1image, 1, 3, 32, 32)

    #print(inputs_image[:,0, :, :, :].reshape(-1, 3, 32, 32).shape)
    joint_prediction,_,  h = model.autoregress_joint_closed(image_input_data[:,0, :, :, :].reshape(-1, 3, 32, 32), joint_input_datum,image_goal_datum, h=None)
    joint_predictions[:, 0, :] = joint_prediction
    for i in range(1, image_input_data.shape[1]):
        joint_prediction,_,  h = model.autoregress_joint_closed(image_input_data[:,i, :, :].reshape(-1, 3, 32, 32), joint_predictions[:, i-1, :], image_goal_datum, h)
        joint_predictions[:, i, :] = joint_prediction

    visualize_joint_test(joint_predictions, joint_target_data, data_num_in_1image ,joint_image_path + "/closed_" + data_name)    

#LSTM : closed-test
def test_predict_joint_closed_LSTM(model, image_data : np.ndarray, joint_data : np.ndarray, model_name : str):
    image_data_tensor = torch.from_numpy(image_data).clone()
    joint_data_tensor = torch.from_numpy(joint_data).clone()

    torch.manual_seed(seed=2)
    data_num = 1
    joint_predictions = torch.zeros((data_num, 79, 14))
    random_index = torch.randint(low=0, high=100, size=(data_num,))
    inputs_image = torch.zeros((data_num, 80, 3, 24, 32))
    inputs_joint = torch.zeros((data_num, 80, 14))

    for i in range(data_num):
        input_image = image_data_tensor[random_index[i]]
        input_joint = joint_data_tensor[random_index[i]]

        inputs_image[i] = input_image 
        inputs_joint[i] = input_joint

    joint_input_datum = inputs_joint[:, 0, :]
    joint_target_data = inputs_joint[:, 1 :, :]
    image_input_data = inputs_image[:, 0:-1, :, :, :]

    print(inputs_image[:,0, :, :, :].reshape(-1, 3, 24, 32).shape)
    
    joint_prediction, h, c = model.autoregress(inputs_image[:,0, :, :, :].reshape(-1, 3, 32, 32), joint_input_datum,  None, None)
    joint_predictions[:, 0, :] = joint_prediction
    for i in range(1, image_input_data.shape[1]):
        joint_prediction, h, c = model.autoregress(inputs_image[:,i, :, :].reshape(-1, 3, 32, 32), joint_predictions[:, i-1, :], h, c)
        joint_predictions[:, i, :] = joint_prediction

    visualize_joint_test(joint_predictions, joint_target_data, data_num, model_name+ "/closed_")  


#open-test -> image : open, joint : open
def test_predict_joint_open(model, image_data : np, joint_data : np, joint_image_path : str, data_name : str, random_index_array : list):

    image_data_narray = image_data.copy()
    joint_data_narray = joint_data.copy()

    data_num_in_1image = 1
    #joint_predictions = torch.zeros((n, 159, 14))
    inputs_image = np.zeros((data_num_in_1image, 80, 3, 32, 32))
    inputs_joint = np.zeros((data_num_in_1image, 80, joint_dim))
    for i in range(data_num_in_1image):
        input_image = image_data_narray[random_index_array[i]]
        joint_datum = joint_data_narray[random_index_array[i]]

        inputs_image[i] = input_image 
        inputs_joint[i] = joint_datum

    joint_input_data = inputs_joint[:, 0:-1, :]
    joint_target_data = inputs_joint[:, 1 :, :] 
    image_inputs_data = inputs_image[:, 0:-1, :, :, :]
    
    #create_goal_image_data_function is (np->np)_function
    image_goal = create_goal_image_data(inputs_image[:, 1:, :, :, :])
    
    joint_input_data = torch.from_numpy(joint_input_data).float()
    joint_target_data = torch.from_numpy(joint_target_data).float()
    image_inputs_data = torch.from_numpy(image_inputs_data).float()
    image_goal = torch.from_numpy(image_goal).float()

    joint_predictions, _, _ = model.forward(image_inputs_data.reshape(-1, 3, 32, 32), joint_input_data, image_goal.reshape(-1, 3, 32, 32))
    
    visualize_joint_test(joint_predictions, joint_target_data, data_num_in_1image, joint_image_path + "/open_" + data_name)





def cal_MSE(joint_targets : np, joint_predictions : np)-> float:
    joint_targets = joint_targets.reshape(-1, 1)
    joint_predictions = joint_predictions.reshape(-1, 1)
    loss = 0

    for i in range(joint_predictions.shape[0]):
        loss = (joint_predictions[i, 0]-joint_targets[i, 0])**2

    return loss/joint_predictions.shape[0]

def cal_max_differency(joint_targets : np, joint_predictions : np)-> float:
    diff_prediction_target = np.abs(joint_predictions - joint_targets)
    
    return np.max(diff_prediction_target) 

def visualize_joint_test(joint_predictions :np, joint_target_data : np, data_num : int, joint_image_path : str):
    joint_predictions = joint_predictions.detach().cpu().numpy()
    joint_target_data = joint_target_data.detach().cpu().numpy()

    legend_label = []
    for i in range(data_num):
        legend_label.append('pred_data{}'.format(i+1))
        legend_label.append('target_data{}'.format(i+1))

    mean_mse_loss = 0
    fig = plt.figure(figsize=(20, 10))
    
    plt.subplots_adjust(hspace=1, wspace=0.6)
    for i  in range(data_num):
        for j in range(joint_dim):
            ax = fig.add_subplot(3, 2, j+1)
            ax.set_xlim(1, 80)
            if i%2 == 0:
                pre_color = 'royalblue'
                target_color = 'deepskyblue'
            else:
                pre_color = 'seagreen'
                target_color = 'mediumspringgreen'
            ax.plot(joint_predictions[i, :, j], "-", color=pre_color)
            ax.plot(joint_target_data[i, :, j], ":", color = target_color)
            ax.legend(legend_label, loc='center left', bbox_to_anchor=(1., .5))

            ax.set_xlabel(f'dim {j+1}...\
                        mae={cal_MSE(joint_target_data[i, :, j], joint_predictions[i, :, j])}\
                        max_diff={cal_max_differency(joint_target_data[i, :, j], joint_predictions[i, :, j])}')
            
            mean_mse_loss += cal_MSE(joint_target_data[i, :, j], joint_predictions[i, :, j])
    plt.suptitle(f"compare_predictions_targets....... mean_mse_loss : {mean_mse_loss/(5*data_num)}")        
    plt.savefig(joint_image_path+"compare_predictions_targets_data") 

