from src.data_processing.data_preprocessing import print_shape_and_maxMin, joint_regularization
#from torchsummary import summary
import torch
import numpy as np
from src.Trainer.helper_train import visualize_loss, EarlyStopping
import torch.nn as nn
from src.model.EtoE import EtoE_RNN
from hyperparameter import *
from src.test.main_tester import test
from src.data_processing.dataset_dataloader import create_dataloader, MyDataset
from src.Trainer.trainer import Model_train
from src.path import mkdir_ofimage_from_modelName

seed = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#seed値固定
if device == 'cuda':
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



def model_train(model, joint_data : str, image_data : str,
                model_name : str)->None:
    early_stopping = EarlyStopping()
    model_train = Model_train(model, model_name, lr, optimizer, batch_size, epochs)
    model = model_train.fit(image_data, joint_data, function_create_dataloader=create_dataloader,
                            early_stopping=early_stopping)
    model_train.visualize_loss(model_train.loss_dict, 'with_weight')
    model_train.visualize_loss(model_train.loss_noweight_dict, 'no_weight')

def model_test(model_name : str,joint_data : str, image_data : str,z_dim : int, test_function)->None:
    model_path = "source/model/learned_model/" + model_name +".pth"

    #読み込むモデルに合わせてRNN, LSTM, GRUのいずれかに変える
    model = EtoE_RNN(model_name, input_size=joint_dim+z_dim*2, hidden_size=hidden_size, num_layer=num_layer ,z_dim = z_dim,kld_weight=kld_weight, device = 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    test_function(model, model_name,  joint_data, image_data.reshape(60, 80, 3, 32, 32), z_dim)


def model_summary(model, model_name : str):
    model_path = "source/model/learned_model/" + model_name +".pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def count_model_params(model):
        params = 0
        for p in model.parameters():
            if p.requires_grad:
                params += p.numel()
        return params

    print(f"num_of_param : {count_model_params(model)}")
    #summary(model, input_size=((3, 24, 32),(1,14)))


def main():
    joint_data = np.load("data_source/joint_two_cup_60.npy")
    image_data = np.load("data_source/image_two_cup_60.npy")/255.0
    joint_data = joint_data[:,:, :]
    print("joint_data.....")
    print_shape_and_maxMin(joint_data)
    print("image_data.....")
    print_shape_and_maxMin(image_data)
    
    joint_regularized_data, _= joint_regularization(joint_data)
    print("joint_data(regularized).....")
    print_shape_and_maxMin(joint_regularized_data)
    
    
    for i in range(len(z_dim)):
        if not is_thismodel_directory_exist:
            print(f'making dirctory{model_name} ........')
            mkdir_ofimage_from_modelName(model_name+str(z_dim[i]))
        
        #モデル定義
        model = EtoE_RNN(model_name+str(z_dim[i]), input_size=joint_dim+z_dim[i]*2, hidden_size=hidden_size,
                        num_layer=num_layer, z_dim =z_dim[i] ,kld_weight=kld_weight ,device = device)


        #やらせたいタスクを選ぶ、やらなくていいタスクはコメントアウト
        model_train(model, joint_regularized_data, image_data, model_name+str(z_dim[i]))
        
        #joint_dataの正規化の際に正規化されてしまうためもう一度loadする
        #To Do : 正規化の関数ないでjoint_dataをcpoy detachする
        joint_data = np.load("data_source/joint_two_cup_60.npy")
        image_data = np.load("data_source/image_two_cup_60.npy")/255.0
        joint_regularized_data, _= joint_regularization(joint_data)
        model_test(model_name+str(z_dim[i]),joint_regularized_data, image_data,z_dim[i], test_function=test)
        model_summary(model, model_name+str(z_dim[i]))
    
    return 0


if __name__ == "__main__":
    main()
