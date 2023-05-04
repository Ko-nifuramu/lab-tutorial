import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from src.data_processing.dataset_dataloader import MyDataset
from hyperparameter import early_stop_epoch

class Model_train():
    def __init__(self, model, model_name, lr,optimizer, batch_size, epochs):
        self.model = model
        self.lr =lr
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_name = model_name
        self.loss_dict ={'train_loss' : [], 'train_vae_loss' : [], 'train_joint_loss' : [],
                    'train_reconst_loss' : [], 'train_kld' : [], 'train_kld_hat' : [],
                    'val_loss' : [], 'val_vae_loss' : [], 'val_joint_loss' : [],
                    'val_reconst_loss' : [], 'val_kld' : [], 'val_kld_hat' : [],}
        
        self.loss_noweight_dict ={'train_loss' : [], 'train_vae_loss' : [], 'train_joint_loss' : [],
                    'train_reconst_loss' : [], 'train_kld' : [], 'train_kld_hat' : [],
                    'val_loss' : [], 'val_vae_loss' : [], 'val_joint_loss' : [],
                    'val_reconst_loss' : [], 'val_kld' : [], 'val_kld_hat' : [],}
        
        self.stop_epoch = 0
        #self.image_path = "../../image/"+self.model_name

    def fit(self, image_data : np, joint_data : np,
            function_create_dataloader,early_stopping):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataloader, val_dataloader = function_create_dataloader()(image_data, joint_data, self.batch_size, MyDataset)
        #torch.manual_seed(seed=seed)

        model = self.model.to(device)
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)

        #勾配クリッピング
        #grad_clip = 1

        for epoch in range(self.epochs):
            
            train_loss_list = [0, 0, 0, 0, 0, 0]
            train_loss_no_weight_list = [0, 0, 0, 0, 0, 0]
            
            for image_inputs, image_targets, joint_inputs, joint_targets, image_goals in train_dataloader:
                model.train()
                image_inputs, image_targets, joint_inputs, joint_targets, image_goals\
                    = image_inputs.to(device), image_targets.to(device), joint_inputs.to(device), joint_targets.to(device), image_goals.to(device)
                    
                # print(f'image_inputs : {image_inputs.shape}')
                # print(f'image_goals : {image_goals.shape}')
                
                image_inputs = image_inputs.reshape(-1, 3, 32, 32)
                image_targets = image_targets.reshape(-1, 3, 32, 32)
                image_goals = image_goals.reshape(-1, 3, 32, 32)
                
                loss_li, _= model.cal_loss(image_inputs, image_targets, joint_inputs, joint_targets, image_goals)
                #print("loss.shpe {}".format(loss.shape))
                optimizer.zero_grad()
                loss_li[0].backward()
                #nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value=grad_clip)
                optimizer.step()    

                model.eval()
                with torch.no_grad():
                    loss_li, loss_noweight_li\
                        = model.cal_loss(image_inputs, image_targets, joint_inputs, joint_targets, image_goals)
                    for i in range(len(loss_li)):
                        train_loss_list[i] += loss_li[i]
                        train_loss_no_weight_list[i] += loss_noweight_li[i]
                    

            model.eval()
            with torch.no_grad():
                val_loss_list = [0, 0, 0, 0, 0, 0]
                val_loss_no_weight_list = [0, 0, 0, 0, 0, 0]
                
                for image_inputs, image_targets, joint_inputs, joint_targets, image_goals in val_dataloader:
                    image_inputs, image_targets, joint_inputs, joint_targets, image_goals\
                        = image_inputs.to(device), image_targets.to(device), joint_inputs.to(device), joint_targets.to(device), image_goals.to(device)
                    image_inputs = image_inputs.reshape(-1, 3, 32, 32)
                    image_targets = image_targets.reshape(-1, 3, 32, 32)
                    image_goals = image_goals.reshape(-1, 3, 32, 32)
                    
                    loss_li, loss_noweight_li\
                        = model.cal_loss(image_inputs, image_targets, joint_inputs, joint_targets, image_goals)
                    for i in range(len(loss_li)):
                        train_loss_list[i] += loss_li[i]
                        train_loss_no_weight_list[i] += loss_noweight_li[i]


            print("Epoch: {}/{} ".format(epoch + 1, self.epochs),
                "Train_loss: {} ".format(train_loss_list[0]/len(train_dataloader)),
                "Train_vae_loss: {} ".format(train_loss_list[1]/len(train_dataloader)),
                "Train_joint_loss: {} ".format(train_loss_list[2]/len(train_dataloader)),
                "Train_reconst_loss : {} ".format(train_loss_list[3]/len(train_dataloader)),
                "Train_kld : {} ".format(train_loss_list[4]/len(train_dataloader)),
                "Train_kld_hat : {} ".format(train_loss_list[5]/len(train_dataloader)),
                "Val_loss: {}".format(val_loss_list[0]/len(val_dataloader)),
                "Val_vae_loss : {}".format(val_loss_list[1]/len(val_dataloader)),
                "Val_joint_loss : {}".format(val_loss_list[2]/len(val_dataloader)),
                "Val_reconst_loss : {} ".format(val_loss_list[3]/len(val_dataloader)),
                "Val_kld : {} ".format(val_loss_list[4]/len(val_dataloader)),
                "Val_kld_hat : {} ".format(val_loss_list[5]/len(val_dataloader)),
                )

            self.loss_dict["train_loss"].append(train_loss_list[0]/len(train_dataloader))
            self.loss_dict["train_vae_loss"].append(train_loss_list[1]/len(train_dataloader))
            self.loss_dict["train_joint_loss"].append(train_loss_list[2]/len(train_dataloader))
            self.loss_dict["train_reconst_loss"].append(train_loss_list[3]/len(train_dataloader))
            self.loss_dict["train_kld"].append(train_loss_list[4]/len(train_dataloader))
            self.loss_dict["train_kld_hat"].append(train_loss_list[5]/len(train_dataloader))
            self.loss_dict["val_loss"].append(val_loss_list [0]/len(val_dataloader))
            self.loss_dict["val_vae_loss"].append(val_loss_list[1]/len(val_dataloader))
            self.loss_dict["val_joint_loss"].append(val_loss_list[2]/len(val_dataloader))
            self.loss_dict["val_reconst_loss"].append(val_loss_list[3]/len(val_dataloader))
            self.loss_dict["val_kld"].append(val_loss_list[4]/len(val_dataloader))
            self.loss_dict["val_kld_hat"].append(val_loss_list[5]/len(val_dataloader))
            
            self.loss_noweight_dict["train_loss"].append(train_loss_no_weight_list[0]/len(train_dataloader))
            self.loss_noweight_dict["train_vae_loss"].append(train_loss_no_weight_list[1]/len(train_dataloader))
            self.loss_noweight_dict["train_joint_loss"].append(train_loss_no_weight_list[2]/len(train_dataloader))
            self.loss_noweight_dict["train_reconst_loss"].append(train_loss_no_weight_list[3]/len(train_dataloader))
            self.loss_noweight_dict["train_kld"].append(train_loss_no_weight_list[4]/len(train_dataloader))
            self.loss_noweight_dict["train_kld_hat"].append(train_loss_no_weight_list[5]/len(train_dataloader))
            self.loss_noweight_dict["val_loss"].append(val_loss_no_weight_list[0]/len(val_dataloader))
            self.loss_noweight_dict["val_vae_loss"].append(val_loss_no_weight_list[1]/len(val_dataloader))
            self.loss_noweight_dict["val_joint_loss"].append(val_loss_no_weight_list[2]/len(val_dataloader))
            self.loss_noweight_dict["val_reconst_loss"].append(val_loss_no_weight_list[3]/len(val_dataloader))
            self.loss_noweight_dict["val_kld"].append(val_loss_no_weight_list[4]/len(val_dataloader))
            self.loss_noweight_dict["val_kld_hat"].append(val_loss_no_weight_list[5]/len(val_dataloader))
            

            if math.isnan(val_loss_list[0]) and epoch>0:
                break
            
            if epoch > early_stop_epoch:
                early_stopping(val_loss_list[0]/len(val_dataloader), model) # 最良モデルならモデルパラメータ保存
                if early_stopping.early_stop: 
                            # 一定epochだけval_lossが最低値を更新しなかった場合、ここに入り学習を終了
                    break
        self.stop_epoch = epoch
        path = "source/Trainer/checkpoint.pt"       
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))       
        path = "source/model/learned_model/"+self.model_name+".pth"
        torch.save(model.to('cpu').state_dict(), path)
        return model
    
    def visualize_loss(self,loss_dict, fig_name : str):
        fig = plt.figure()
        ax1 = fig.add_subplot()
        ax1.plot(loss_dict['train_vae_loss'], linestyle="solid")
        ax1.plot(loss_dict['train_joint_loss'], linestyle="dashed")
        ax1.plot(loss_dict['train_loss'], linestyle = "dotted")
        ax1.plot(loss_dict['val_vae_loss'], linestyle="solid")
        ax1.plot(loss_dict['val_joint_loss'], linestyle="dashed")
        ax1.plot(loss_dict['val_loss'], linestyle = "dotted")
        ax1.set_yscale('log')
        ax1.legend(['train_vae_loss * vae_weight', 'train_joint_loss * joint_weight', 'train_loss', 'val_vae_loss * vae_weight', 'val_joint_loss * joint_weight', 'val_loss'],\
                loc='center left', bbox_to_anchor=(1., .5))
        

        ax1.set_xlim(0, self.stop_epoch)
        ax1.set_title('each_loss')
        fig.savefig("image/"+self.model_name + "/" + "Training_loss", bbox_inches='tight')

        fig = plt.figure()
        ax2 = fig.add_subplot()
        ax2.plot(loss_dict['train_reconst_loss'], linestyle="solid")
        ax2.plot(loss_dict['train_kld'], linestyle="dashed")
        ax2.plot(loss_dict['train_kld_hat'], linestyle = "dotted")
        ax2.plot(loss_dict['val_reconst_loss'], linestyle="solid")
        ax2.plot(loss_dict['val_kld'], linestyle="dashed")
        ax2.plot(loss_dict['val_kld_hat'], linestyle = "dotted")
        ax2.set_yscale('log')
        ax2.legend(['train_reconst_loss *scale_adjust/(self.var*2)', 'train_kld', 'train_kld_hat', 'val_reconst_loss * scale_adjust/(self.var*2)', 'weight_kld * val_kld', 'weight_kld_hat * val_kld_hat'],\
                loc='center left', bbox_to_anchor=(1., .5))
        

        ax2.set_xlim(0, self.stop_epoch)
        ax2.set_title('each_loss(vae_weight)')
        fig.savefig("image/"+self.model_name + "/" + "Training_vae_loss"+fig_name, bbox_inches='tight')

