import argparse
import oxford_pet
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import utils
import os
from models import unet, resnet34_unet

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='UNet.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, default=os.path.join('C:\programming\Python\deep_learning\Lab3'),help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = oxford_pet.load_dataset(os.path.join(args.data_path, 'dataset', 'oxford-iiit-pet'), mode='test')
    dataloader = DataLoader(dataset, batch_size=args.batch_size) 
    model = torch.load(os.path.join(args.data_path, 'saved_models', args.model), map_location=device)
    model.eval()
    score = 0
    number = 0
    for sample in tqdm(dataloader, desc='inference', leave=False):
        input = sample["image"]
        input = input.float().to(device)
        label = sample["mask"].to(device)
        outputs = model(input)
        outputs = torch.where(outputs < 0.5, torch.tensor(0., device=device), torch.tensor(1., device=device))
        score += utils.dice_score(outputs, label)
        number += 1
    print(f'Dice score: {score/number:.4f}')
