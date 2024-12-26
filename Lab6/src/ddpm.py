import torch
from torch import nn
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
import iclevr
from torchvision import transforms
import ddpm_model
import os

batch_size = 16
lr=0.0002
num_train_timesteps = 1000
accum_grad=8
num_epochs = 101
save_epochs = 5

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
image_dir = './iclevr'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = iclevr.IclevrDataset(image_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule='squaredcos_cap_v2')

net = ddpm_model.ClassConditionedUnet().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)

net.train()
for epoch in range(num_epochs):
    accum=0
    losses=0
    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, num_train_timesteps-1, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
        pred = net(noisy_x, timesteps, y)
        loss = loss_fn(pred, noise)
        losses+=loss.item()
        loss=loss/accum_grad
        loss.backward()
        accum+=1
        if accum % accum_grad == 0:
            optimizer.step()
            optimizer.zero_grad()
        
    scheduler.step()

    print(f'Finished epoch {epoch}. loss values: {losses:03f}')
    if epoch % save_epochs == 0:
        torch.save(net.state_dict(), os.path.join("./DDPMcheckpoints", f"ddpm epoch={epoch} loss={losses:03f}.pth"))
        print(f"save epoch={epoch} loss={losses:03f}.pth")


