import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import random
import math


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))
    def save_transformer_checkpoint(self, epoch):
        torch.save(self.transformer.state_dict(), os.path.join("./transformer_checkpoints", f"epoch={epoch}.ckpt"))
        print(f'save ckpt, epoch={epoch}')

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        _,zq,_ = self.vqgan.encode(x)
        zq = zq.reshape(-1,256)
        return zq
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine", t=0, T=8):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return 1-t/T
        elif mode == "cosine":
            return math.cos(math.pi/2*(t/T))
        elif mode == "square":
            return 1-(t/T)**2
        else:
            return random.random()*0.4 + 0.1

##TODO2 step1-3:            
    def forward(self, x):
        z_indices=None #ground truth
        logits = None  #transformer predict the probability of tokens
        z_indices_ori=self.encode_to_z(x)
        z_indices = z_indices_ori.clone()
        mask = torch.rand(z_indices.shape) < self.gamma_func(mode='train')
        z_indices[mask]=1024
        logits=self.transformer(z_indices)
        return logits, z_indices_ori

##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask_b, mask_num, ratio, device):
        mask_bc=mask_b.clone()
        z_indices_ori=z_indices.clone()
        z_indices = torch.where(mask_b == True, 1024, z_indices)
        logits = self.transformer(z_indices)

        #Apply softmax to convert logits into a probability distribution across the last dimension.
        softmax=nn.Softmax(dim=-1)
        logits = softmax(logits)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(logits, dim=-1)

        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = torch.rand(z_indices_predict_prob.shape).to(device)
        gumbel_noise = -torch.log(-torch.log(g))*1e-3
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * gumbel_noise
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        confidence = torch.where(mask_b == False, float('inf'), confidence)

        #sort the confidence for the rank 
        sort_confidence, sort_indices = torch.sort(confidence)

        #define how much the iteration remain predicted tokens by mask scheduling
        remain_mask=int(mask_num*ratio)
        remain_mask_index = sort_indices[:,:remain_mask]
        mask_bc[0,remain_mask_index]=True
        
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        not_mask_index = sort_indices[:,remain_mask:]
        z_indices_predict=torch.where(mask_b == False, z_indices_ori, z_indices_predict)
        mask_bc[0,not_mask_index]=False
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
