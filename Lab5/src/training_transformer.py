import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim = self.configure_optimizers(args.learning_rate)
        self.prepare_training()
 
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader, device):
        self.model.to(device)
        self.model.vqgan.eval()
        self.model.transformer.train()
        criterion = nn.CrossEntropyLoss()
        total_loss=0
        accum=0
        for img in tqdm(train_loader):
            img = img.to(device)
            logits, z_indices = self.model.forward(img)
            logits_flat = logits.view(-1, 1025)
            z_indices_flat = z_indices.view(-1)
            loss = criterion(logits_flat, z_indices_flat)
            total_loss+=loss.item()
            loss=loss/args.accum_grad
            loss.backward()
            accum+=1
            if accum % args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()

        print(f"Total Loss: {total_loss}")

    def eval_one_epoch(self, val_loader, device):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0

        with torch.no_grad():
            for img in tqdm(val_loader):
                img = img.to(device)
                logits, z_indices = self.model.forward(img)
                logits_flattened = logits.view(-1, 1025)
                z_indices_flat = z_indices.view(-1)
                loss = criterion(logits_flattened, z_indices_flat)
                total_loss += loss.item()

        print(f"Validation Loss: {total_loss}")

    def configure_optimizers(self, lr):
        optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5], gamma=0.1)
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    train_transformer.prepare_training()
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_transformer.train_one_epoch(train_loader, args.device)
        train_transformer.eval_one_epoch(val_loader, args.device)
        if epoch % args.save_per_epoch == 0:
            train_transformer.model.save_transformer_checkpoint(epoch)