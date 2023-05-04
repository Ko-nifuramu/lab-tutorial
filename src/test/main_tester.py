from source.test.joint.joint_prediction import*
from source.test.test_helper import get_model_from_modelName, test_joint_prediction_closed
from source.test.latent.latent_pca import cal_task_mean_latent, visualize_2task_pca
from source.test.latent.latent_hat_pca import cal_task_mean_next_latent, visualize_nextlatent_2task_pca
from source.test.latent.each_latent_dim import visualize_randomly_task_latent
from source.test.image_reconst.generate_reconst import generate_xy_Image
import os
from source.path import get_modelPath_from_modelName
from source.test.joint.predict_lots_data import joint_predictions_lots_data
from hyperparameter import batch_size, is_thismodel_directory_exist



def test(model, model_name:str, joint_data_regularized:np, image_data:np, z_dim : int)->None:

    if not is_thismodel_directory_exist:
        print(f'making dirctory{model_name} ........')
        mkdir_ofimage_from_modelName(model_name + "/joint_prediction")

    joint_image_path = "image/" + model_name + "/joint_prediction"

    generate_xy_Image(model, image_data=image_data, joint_data=joint_data_regularized ,model_name=model_name)

    # #joint_test
    # test_joint_prediction_closed(model,image_data, joint_data_regularized, model_name)
    # test_predict_joint_open(model, image_data, joint_data_regularized, model_name)

    #latent_test
    left_pca, right_pca, explained_variance_ratio = cal_task_mean_latent(model, image_data)
    visualize_2task_pca(left_pca, right_pca, explained_variance_ratio,model_name)
    left_hat_pca, right_hat_pca, explained_variance_ratio = cal_task_mean_next_latent(model, image_data, joint_data_regularized)
    visualize_nextlatent_2task_pca(left_hat_pca, right_hat_pca, explained_variance_ratio,model_name)

    visualize_randomly_task_latent(model, image_data,model_name, z_dim)

    #joint_test
    joint_predictions_lots_data(model, joint_data_regularized, image_data, batch_size, joint_image_path)

