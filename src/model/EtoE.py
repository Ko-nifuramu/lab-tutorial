from src.model.RNN import *
from src.model.vae import Vision_VAE
from src.model.model_helper import get_RNNmodel_from_modelName
from hyperparameter import  kld_hat_weight



class EtoE_RNN(nn.Module):
    def __init__(self, model_name, input_size,  hidden_size, num_layer, z_dim, kld_weight, device):
        super(EtoE_RNN, self).__init__()
        self.z_dim = z_dim
        self.vision_vae = Vision_VAE(z_dim, device)
        self.rnn = get_RNNmodel_from_modelName(model_name, input_size, hidden_size, num_layer, z_dim)
        
        self.beta = kld_weight
        self.beta_hat = kld_hat_weight
        self.var = 1e-2
        self.weight_vae = vae_weight
        self.weight_joint = joint_weight
        
    
    #z_latent, joint -> open
    def forward(self, i_t, j_pre, i_g):
        z_i_t, y_t = self.vision_vae.forward(i_t)
        z_i_g, y_g = self.vision_vae.forward(i_g)
        mean_hat, log_var_hat, j_out = self.rnn.forward(z_i_t,z_i_g, j_pre)
        z_latent_next = self.vision_vae.reparameterize(mean_hat, log_var_hat)
        
        j_next = torch.tanh(j_out)
        recon_image_next = self.vision_vae.decoder(z_latent_next.reshape(-1, self.z_dim))
        
        return j_next, recon_image_next, mean_hat

    #z_latent -> open, joint -> closed
    def autoregress_joint_closed(self, i_t, j_pre, i_g, h):
        z_i_t, y_t = self.vision_vae.forward(i_t)
        z_i_g, y_g = self.vision_vae.forward(i_g)
        mean_hat, log_var_hat, j_out, h = self.rnn.autoregress(z_i_t,z_i_g, j_pre, h)
        z_latent_next = self.vision_vae.reparameterize(mean_hat, log_var_hat)
        
        j_next = torch.tanh(j_out)
        recon_image_next = self.vision_vae.decoder(z_latent_next.reshape(-1, self.z_dim))
        
        return j_next, recon_image_next, h    

    def cal_loss(self, i ,i_target, j_pre, j_target,i_g, criterion = nn.MSELoss()):

        loss_list = []
        loss_noweight_list = []
        
        scale_adjust = i.shape[1] * i.shape[2] * i.shape[3]/ self.z_dim
        
        #vae encoder, reparameterize
        mean, log_var = self.vision_vae.encoder(i)
        z_i = self.vision_vae.reparameterize(mean, log_var)
        mean_g, log_var_g = self.vision_vae.encoder(i_g)
        z_i_g = self.vision_vae.reparameterize(mean_g, log_var_g)
        
        #RNN
        mean_hat, log_var_hat, j_out = self.rnn.forward(z_i, z_i_g, j_pre)
        z_latent_next = self.vision_vae.reparameterize(mean_hat, log_var_hat)
        
        j_next = torch.tanh(j_out)
        prediction_image_next = self.vision_vae.decoder(z_latent_next.reshape(-1, self.z_dim))


        #calculate losses
        loss_joint = criterion(j_target, j_next)
        image_recon = criterion(i_target, prediction_image_next)
        
        kld = -torch.mean(1+log_var- mean**2 - torch.exp(log_var))/2
        
        mean_hat = mean_hat.reshape(-1, self.z_dim)
        log_var_hat = log_var_hat.reshape(-1, self.z_dim)
        kld_hat = torch.mean(-1+(log_var_hat-log_var)+(torch.exp(log_var) + (mean - mean_hat) ** 2) / torch.exp(log_var_hat))/2
        loss_vae = image_recon*scale_adjust/(self.var*2) + self.beta*kld + self.beta_hat*kld_hat
        
        loss = self.weight_vae *loss_vae + self.weight_joint*loss_joint
        
        loss_list.extend(loss, self.weight_vae *loss_vae, self.weight_joint*loss_joint, image_recon*scale_adjust/(self.var*2), self.beta*kld, self.beta_hat*kld_hat)
        loss_noweight_list.extend(image_recon+kld+kld_hat+loss_joint, image_recon+kld+kld_hat, loss_joint, image_recon, kld, kld_hat)

        
        return loss_list, loss_noweight_list