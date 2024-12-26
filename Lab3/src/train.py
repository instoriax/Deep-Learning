import os
import argparse
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from models import unet, resnet34_unet
import evaluate
import oxford_pet
import csv
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default=os.path.join('C:\programming\Python\deep_learning\Lab3'), help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--model_name', '-m', type=str, default='UNet')

    return parser.parse_args()
 
if __name__ == "__main__":

    args = get_args()

    if args.model_name == 'UNet':
        model = unet.UNet()
    elif args.model_name == 'ResNet34_UNet':
        model = resnet34_unet.ResNet34_UNet()
    else:
        model = unet.UNet()

    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    train_scores = []
    valid_scores = []
    for epoch in range(args.epochs):
        dataset = oxford_pet.load_dataset(os.path.join(args.data_path, 'dataset', 'oxford-iiit-pet'), mode='train')
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) 
        for sample in tqdm(dataloader, desc='train', leave=False):
            input = sample["image"]
            input = input.float().to(device)
            label = sample["mask"].to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = nn.MSELoss()(output, label)
            loss.backward()
            optimizer.step()
        train_score=evaluate.evaluate(model, args, 'train', device)
        valid_score=evaluate.evaluate(model, args, 'valid', device)
        train_scores.append(train_score)
        valid_scores.append(valid_score)
        print(f'Epoch [{epoch+1}/{args.epochs}], train dice score: {train_score:.4f}, valid dice score: {valid_score:.4f}')

    # with open(args.model_name+'_train.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Epoch', 'Dice score'])
    #     for epoch, acc in enumerate(train_acc, start=1):
    #         writer.writerow([epoch, acc])

    # with open(args.model_name+'_vaild.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Epoch', 'Dice score'])
    #     for epoch, acc in enumerate(valid_acc, start=1):
    #         writer.writerow([epoch, acc])

    # torch.save(model, args.model_name+'.pth')

    

    # model = torch.load(os.path.join(args.data_path, 'saved_models', 'UNet.pth'), map_location=device)
    # model.to(device)
    # UNet_train_score=evaluate.evaluate(model, args, 'train', device)
    # UNet_valid_score=evaluate.evaluate(model, args, 'valid', device)
    # UNet_test_score=evaluate.evaluate(model, args, 'test', device)

    # model = torch.load(os.path.join(args.data_path, 'saved_models', 'ResNet34_UNet.pth'), map_location=device)
    # model.to(device)
    # ResNet34_UNet_train_score=evaluate.evaluate(model, args, 'train', device)
    # ResNet34_UNet_valid_score=evaluate.evaluate(model, args, 'valid', device)
    # ResNet34_UNet_test_score=evaluate.evaluate(model, args, 'test', device)

    # print('----------UNet----------')
    # print(f'UNet             | Train score: {UNet_train_score:7.2f} | Valid score: {UNet_valid_score:7.2f} | Test score: {UNet_test_score:7.2f}')
    # print('----------ResNet34_UNet----------')
    # print(f'ResNet34_UNet    | Train score: {ResNet34_UNet_train_score:7.2f} | Valid score: {ResNet34_UNet_valid_score:7.2f} | Test score: {ResNet34_UNet_test_score:7.2f}')

