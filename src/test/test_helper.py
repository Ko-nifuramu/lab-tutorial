from source.model.RNN import*
from hyperparameter import hidden_size, num_layer
from source.test.joint.joint_prediction import *
import os


#Attention : 新しいモデルを作成したら分岐を増やす
def get_model_from_modelName(model_name : str, z_dim : int):
    print(model_name)
    if "RNN" in model_name:
        return RNN_(input_size=14+z_dim, hidden_size=hidden_size,
                            num_layers=num_layer ,z_dim = z_dim)
    elif "LSTM" in model_name:
        return LSTM_(input_size=14+z_dim, hidden_size=hidden_size,
                            num_layers=num_layer ,z_dim = z_dim)
    elif "GRU" in model_name:
        return GRU_(input_size=14+z_dim, hidden_size=hidden_size,
                            num_layers=num_layer ,z_dim = z_dim)
    else:
        raise ValueError("enter accurate model_path")
    
def test_joint_prediction_closed(model,image_data, joint_data, model_name):
    if "LSTM" in model_name:
        test_predict_joint_closed_LSTM(model, image_data, joint_data, model_name)
    else:
        test_predict_joint_closed_RNN(model, image_data, joint_data, model_name)

