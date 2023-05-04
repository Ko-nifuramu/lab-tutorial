import os

def get_modelPath_from_modelName(model_name : str):
    return 'source/model/learned_model/'+model_name+'.pth'

def get_modelName_form_modelPath(model_path : str):
    return model_path[27:]-'.pth'


def mkdir_ofimage_from_modelName(model_name : str) -> str:
    path = "image/"+str(model_name)
    try:
        os.makedirs(path)
    except OSError as e:
        print('this dirctory is already made.')