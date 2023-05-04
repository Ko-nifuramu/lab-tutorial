import numpy as np
import torch
import matplotlib.pyplot as plt
from src.data_processing.data_preprocessing import print_shape_and_maxMin
from src.data_processing.dataset_dataloader import create_goal_image_data


#x -> (encoder) -> z_latent ->(RNN)-> z_latent_hat -> (decoder) -> x_hat
#16000のデータ化から、ランダムにデータxを１０個生成してx,x_hatを比較
def generate_xy_Image(model, image_data : np.ndarray, joint_data : np.ndarray, model_name : str):
    device = 'cpu'
    image_data = image_data.reshape(-1, 3, 32, 32)
    image_data_narray = image_data.copy()
    joint_data_narray = joint_data.copy()
    
    inputs, outputs = genrate_random_inputAndReconst(model, image_data_narray, joint_data_narray)
    
    #visualize x(input) images
    input_images =inputs.to(device).detach().numpy().copy()
    print("input_image.....")
    print_shape_and_maxMin(input_images)
    for i, image in enumerate(input_images):
        plt.title("original")
        plt.subplot(2, 5, i+1)
        plt.imshow(image.reshape(32, 32, 3))
        plt.axis("off")

    
    plt.savefig("image/"+model_name+"/input_images.png")

    #visualize x_hat images
    output_images = (outputs).to(device).detach().numpy().copy()
    print("renconst_image")
    print_shape_and_maxMin(output_images)
    for i, image in enumerate(output_images):
        plt.title("reconst")
        plt.subplot(2, 5, i+1)
        image = image.reshape(32, 32, 3)
        plt.imshow(image)
        plt.axis("off")

    #plt.tight_layout()
    plt.savefig("image/" + model_name + "/reconst_images.png")     

def genrate_random_inputAndReconst(model, image_data_narray : np, joint_data_narray : np):
    device =  "cpu"
    image_data_narray = image_data_narray.reshape(-1, 160, 3, 32, 32)
    
    inputs = np.zeros((10, 3, 32, 32))
    joint_inputs = np.zeros((10, 5))
    outputs = np.zeros((10, 3, 32, 32))
    vision_goal_data = np.zeros((10, 3, 32, 32))

    #pick input_imgags up randomly 
    np.random.seed(seed=2)
    random_data_index = np.random.randint(low=0, high=image_data_narray.shape[0], size=(10,))
    random_step_index = np.random.randint(low=0, high=image_data_narray.shape[1]-1, size=(10,))
    
    for i in range(10):
        input_data = image_data_narray[random_data_index[i]]
        # print(f'input_data : {input_data.shape}')
        input_joint_data = joint_data_narray[random_data_index[i]]#(160, 5)
        input_goal_data = create_goal_image_data(input_data)#[0](1, 3, 32, 32)
        inputs[i] = input_data[random_step_index[i]]#(1, 3, 32, 32)
        joint_inputs[i] = input_joint_data[i]
        vision_goal_data[i] = input_goal_data
    
    inputs = torch.from_numpy(inputs).float()
    joint_inputs = torch.from_numpy(joint_inputs).float()
    vision_goal_data = torch.from_numpy(vision_goal_data).float()
    
    #calculate x_hat   
    model.eval()
    with torch.no_grad():
        model = model.to(device)
        inputs = inputs.to(device)
        _, outputs,_ = model.forward(inputs, joint_inputs, vision_goal_data)

    return inputs, outputs 
