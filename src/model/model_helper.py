from source.model.RNN import *

#Attention : 新しいモデルを作成したら分岐を増やす
def get_RNNmodel_from_modelName(model_name,input_size, hidden_size, num_layer, z_dim):
    if "RNN" in model_name:
        return RNN_(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layer ,z_dim = z_dim)
    elif "LSTM" in model_name:
        return LSTM_(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layer ,z_dim = z_dim)
    elif "GRU" in model_name:
        return GRU_(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layer ,z_dim = z_dim)
    else:
        raise ValueError("enter accurate model_path")