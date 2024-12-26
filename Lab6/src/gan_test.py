import torch
import os
import json
import numpy as np
from torchvision.utils import save_image
import gan_model
import evaluator
import torchvision.utils

nc = 3
nz = 100
ngf = 64
num_classes = 24
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


generator_path = './GANcheckpoints/checkpointG.pth'
netG = gan_model.Generator(nz, ngf, nc, num_classes).to(device)
netG.load_state_dict(torch.load(generator_path, map_location=device))
netG.eval()

with open('./json/test.json', 'r') as f:
    test_labels_json = json.load(f)

# with open('./json/new_test.json', 'r') as f:
#     test_labels_json = json.load(f)


def labels_to_one_hot(labels, label_to_index, num_classes):
    one_hot = np.zeros(num_classes, dtype=np.float32)
    for label in labels:
        index = label_to_index[label]
        one_hot[index] = 1.0
    return one_hot


label_to_index = {
    "gray cube": 0, "red cube": 1, "blue cube": 2, 
    "green cube": 3, "brown cube": 4, "purple cube": 5, 
    "cyan cube": 6, "yellow cube": 7, "gray sphere": 8, 
    "red sphere": 9, "blue sphere": 10, "green sphere": 11, 
    "brown sphere": 12, "purple sphere": 13, "cyan sphere": 14, 
    "yellow sphere": 15, "gray cylinder": 16, "red cylinder": 17, 
    "blue cylinder": 18, "green cylinder": 19, "brown cylinder": 20, 
    "purple cylinder": 21, "cyan cylinder": 22, "yellow cylinder": 23}
num_classes = 24


test_labels = np.array([labels_to_one_hot(labels, label_to_index, num_classes) for labels in test_labels_json])
test_labels = torch.tensor(test_labels, dtype=torch.float32).to(device)


num_test_samples = len(test_labels)


noise = torch.randn(num_test_samples, nz, 1, 1, device=device)
with torch.no_grad():
    fake_images = netG(noise, test_labels)
evaluation_model = evaluator.evaluation_model()
acc=evaluation_model.eval(fake_images, test_labels)
print(acc)

output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
std = torch.tensor(std).view(1, 3, 1, 1).to(device)
fake_images = fake_images * std + mean


output=torchvision.utils.make_grid(fake_images)
save_image(output, os.path.join(output_dir, f'test_image.png'))

