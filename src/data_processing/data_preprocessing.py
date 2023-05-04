import numpy as np

def print_shape_and_maxMin(data : np)->None:
    print(f'data.shape : {data.shape}')
    print(f'data_max : {np.max(data)}')
    print(f'data_min : {np.min(data)}')


def joint_regularization(joint_data : np):
    joint_max_and_min_eachdim_array = []
    
    joint_data_reshape= joint_data.reshape(-1,joint_data.shape[-1])
    joint_data_regurarized_2dim = np.zeros_like(joint_data_reshape)

    for i in range(joint_data_reshape.shape[1]):
        joint_data_max = np.max(joint_data_reshape[:, i])
        joint_data_min = np.min(joint_data_reshape[:, i])
        
        joint_max_and_min_eachdim_array.append([joint_data_max, joint_data_min])
        
        joint_data_reshape[:, i] = joint_data_reshape[:, i] - joint_data_min
        joint_data_reshape[:, i] *= 2/(joint_data_max - joint_data_min)
        joint_data_regurarized_2dim[:, i] = joint_data_reshape[:, i] - 1#data max:1, min:-1
    
    
    joint_regularized_data = joint_data_regurarized_2dim.reshape(joint_data.shape)
    
    return joint_regularized_data, joint_max_and_min_eachdim_array

#ロボットを動かす時なので、data 1個分を入力する時を想定している
def reverse_joint_regurlarization(joint_data_regurarized_1data : np, joint_max_and_min_each_dim : float)->np:
    '''
    expected shape
    
    joint_data_regularized.shape -> (1, joint_dim)
    joint_max_and_min_data -> (2 ,joint_dim)
    '''
    joint_data_reverse = joint_data_regurarized_1data+ 1
    for i in range(joint_data_regurarized_1data.shape[-1]):
        joint_data_reverse[:, i] = joint_data_reverse[:, i] * (joint_max_and_min_each_dim[0] - joint_max_and_min_each_dim[1])/2
        joint_data_reverse[:, i] = joint_data_reverse[:, i] + joint_max_and_min_each_dim[1]
    

    return joint_data_reverse