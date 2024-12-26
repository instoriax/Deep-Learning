import VGG19
import ResNet50
from dataloader import BufferflyMothLoader
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import csv

batch_size = 16
learning_rate = 0.001
num_epochs = 60
root='Python/deep_learning/Lab2'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(model, mode):
    dataset = BufferflyMothLoader(root=root, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size) 
    model.eval()
    num_samples = 0
    num_correct = 0
    for inputs, labels in tqdm(dataloader, desc='eval-'+mode, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs = outputs.argmax(dim=1)
        num_samples += labels.size(dim=0)
        num_correct += (outputs == labels).sum()
        accuracy = ((num_correct / num_samples) * 100).item()
    return accuracy

def test(model):
    dataset = BufferflyMothLoader(root=root, mode='test')
    dataloader = DataLoader(dataset, batch_size=1)
    model.eval()
    num_samples = 0
    num_correct = 0
    for inputs, labels in tqdm(dataloader, desc='test', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs = outputs.argmax(dim=1)
        num_samples += labels.size(dim=0)
        num_correct += (outputs == labels).sum()
        accuracy = ((num_correct / num_samples) * 100).item()
    return accuracy

def train(model, optimizer):
    dataset = BufferflyMothLoader(root=root, mode='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    model.train()
    for inputs, labels in tqdm(dataloader, desc='train', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

def highest_accuracy():
    model = torch.load(root+'/VGG19.pth', map_location=device)
    model.to(device)
    VGG19_train_acc=evaluate(model, 'train')
    VGG19_acc=test(model)

    model = torch.load(root+'/ResNet50.pth', map_location=device)
    model.to(device)
    ResNet50_train_acc=evaluate(model, 'train')
    ResNet50_acc=test(model)

    print('----------VGG19----------')
    print(f'VGG19         |   Train accuracy: {VGG19_train_acc:7.2f}%|   Test accuracy: {VGG19_acc:7.2f}%')

    print('----------ResNet50----------')
    print(f'ResNet50      |   Train accuracy: {ResNet50_train_acc:7.2f}%|   Test accuracy: {ResNet50_acc:7.2f}%')

if __name__ == "__main__":
    # model = VGG19.VGG19()
    # model_name = 'VGG19'

    model = ResNet50.ResNet50()
    model_name = 'ResNet50'
    
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) 

    train_acc = []
    valid_acc = []
    for epoch in range(num_epochs):
        train(model, optimizer)
        train_accuracy=evaluate(model, 'train')
        valid_accuracy=evaluate(model, 'valid')
        train_acc.append(train_accuracy)
        valid_acc.append(valid_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], train accuracy: {train_accuracy:.4f}%, valid accuracy: {valid_accuracy:.4f}%')

    with open(model_name+'_train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Accuracy'])
        for epoch, acc in enumerate(train_acc, start=1):
            writer.writerow([epoch, acc])

    with open(model_name+'_vaild.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Accuracy'])
        for epoch, acc in enumerate(valid_acc, start=1):
            writer.writerow([epoch, acc])
    torch.save(model, root+'/'+model_name+'.pth')


    highest_accuracy()


