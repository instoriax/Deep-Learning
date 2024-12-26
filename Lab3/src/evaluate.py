import oxford_pet
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import utils
import os

def evaluate(model, args, mode, device):
    dataset = oxford_pet.load_dataset(os.path.join(args.data_path, 'dataset', 'oxford-iiit-pet'), mode=mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size) 
    model.eval()
    score = 0
    number = 0
    for sample in tqdm(dataloader, desc='eval-'+mode, leave=False):
        input = sample["image"]
        input = input.float().to(device)
        label = sample["mask"].to(device)
        outputs = model(input)
        outputs = torch.where(outputs < 0.5, torch.tensor(0., device=device), torch.tensor(1., device=device))
        score += utils.dice_score(outputs, label)
        number += 1
    return (score/number)