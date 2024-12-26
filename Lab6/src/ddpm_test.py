import torch
import torchvision
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
import evaluator
import ddpm_model
import os
import json
import numpy as np
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
generator_path = './DDPMcheckpoints/checkpoint.pth'
net = ddpm_model.ClassConditionedUnet().to(device)
net.load_state_dict(torch.load(generator_path, map_location=device))
net.eval()

# with open('./json/test.json', 'r') as f:
#     test_labels_json = json.load(f)

with open('./json/new_test.json', 'r') as f:
    test_labels_json = json.load(f)

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

test_labels = test_labels = np.array([labels_to_one_hot(labels, label_to_index, num_classes) for labels in test_labels_json])
test_labels = torch.tensor(test_labels, dtype=torch.float32).to(device)

x = torch.randn(32, 3, 64, 64).to(device)
evaluation_model = evaluator.evaluation_model()
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
std = torch.tensor(std).view(1, 3, 1, 1).to(device)

denoising=None

for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

    with torch.no_grad():
        residual = net(x, t, test_labels)

    x = noise_scheduler.step(residual, t, x).prev_sample

    if (i+1)%100==0:
        denoising_images = x[0] * std + mean
        if denoising==None:
            denoising=denoising_images
        else:
            denoising=torch.cat((denoising, denoising_images), 0)

        acc=evaluation_model.eval(x, test_labels)
        print(acc)
        images = x * std + mean
        output=torchvision.utils.make_grid(images)
        save_image(output, os.path.join(output_dir, f'DDPMtest_image{i+1}.png'))


denoising=torchvision.utils.make_grid(denoising)
save_image(denoising, os.path.join(output_dir, f'denoising process.png'))