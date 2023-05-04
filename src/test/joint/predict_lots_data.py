import numpy as np
import torch
from hyperparameter import trainsample_ratio
from src.test.joint.joint_prediction import test_predict_joint_closed_RNN, visualize_joint_test, cal_MSE, cal_max_differency, test_predict_joint_open

def joint_predictions_lots_data(model : str, joint_regularized_data : np.ndarray, image_data : np.ndarray, batch_size : int, joint_name_path : str):

    train_image_data, train_joint_input_data, val_image_data, val_joint_input_data\
    = devide_train_val_data(image_data, joint_regularized_data)

    data_count = 1
    data_num_in_1image = 1#一つのグラフに表示させるデータ数

    #train_dataでのテスト
    for data_count in range(10):
        random_index_array = torch.randint(low=0, high=train_joint_input_data.shape[0], size=(data_num_in_1image,))
        test_predict_joint_closed_RNN(model, train_image_data, train_joint_input_data, joint_name_path, "train_data"+str(data_count+1), random_index_array)
        test_predict_joint_open(model, train_image_data, train_joint_input_data, joint_name_path, "train_data"+str(data_count+1), random_index_array)
    
    #validation_dataでのテスト
    for data_count in range(10):
        random_index_array = torch.randint(low=0, high=val_joint_input_data.shape[0], size=(1,))
        test_predict_joint_closed_RNN(model, val_image_data, val_joint_input_data, joint_name_path, "val_data"+str(data_count+1), random_index_array)
        test_predict_joint_open(model, val_image_data, val_joint_input_data, joint_name_path, "train_data"+str(data_count+1), random_index_array)



def devide_train_val_data(image_data : np.ndarray, joint_data : np.ndarray) -> np.ndarray:
    ratio = trainsample_ratio
    indeces = [int(image_data.shape[0] * n) for n in [ratio/2, 0.5, (1+ratio)/2]]
    train_image_data1, val_image_data1, train_image_data2, val_image_data2 = np.split(image_data, indeces, axis=0)
    train_image_data = np.concatenate([train_image_data1, train_image_data2], axis=0)
    val_image_data = np.concatenate([val_image_data1, val_image_data2], axis=0)

    train_joint_input1, val_joint_input1, train_joint_input2, val_joint_input2 = np.split(joint_data, indeces, axis=0)
    train_joint_input_data = np.concatenate([train_joint_input1, train_joint_input2], axis=0)
    val_joint_input_data = np.concatenate([val_joint_input1, val_joint_input2], axis=0)


    return train_image_data, train_joint_input_data, val_image_data, val_joint_input_data