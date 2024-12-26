import torch
import torch.nn as nn
import iclevr
from torchvision import transforms
from tqdm import tqdm
import os
import gan_model
import evaluator


batch_size = 2
iter = 18009/batch_size
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 1000
save_epochs = 1
lr = 0.00001
beta1 = 0.5
num_classes = 24
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




netG = gan_model.Generator(nz, ngf, nc, num_classes).to(device)
netD = gan_model.Discriminator(nc, ndf).to(device)

G_path = './GANcheckpoints/checkpointG.pth'
netG.load_state_dict(torch.load(G_path, map_location=device))
D_path = './GANcheckpoints/checkpointD.pth'
netD.load_state_dict(torch.load(D_path, map_location=device))


criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

# netG.apply(weights_init)
# netD.apply(weights_init)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

image_dir = './iclevr'
dataset = iclevr.IclevrDataset(image_dir, transform=transform)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
evaluation_model = evaluator.evaluation_model()

netD.train()
netG.train()

best=0.4
for epoch in range(num_epochs):
    loss=0
    total_acc=0

    for real_images, one_hot_labels in tqdm(dataloader):
        netD.zero_grad()
        one_hot_labels = one_hot_labels.to(device)
        real_images = real_images.to(device)
        label_real = torch.full((one_hot_labels.size(0),), 1, dtype=torch.float, device=device)
        output = netD(real_images, one_hot_labels).view(-1)
        lossD_real = criterion(output, label_real)
        lossD_real.backward()

        noise = torch.randn(one_hot_labels.size(0), nz, 1, 1, device=device)
        fake_images = netG(noise, one_hot_labels)
        label_fake = torch.full((one_hot_labels.size(0),), 0, dtype=torch.float, device=device)
        output = netD(fake_images.detach(), one_hot_labels).view(-1)
        lossD_fake = criterion(output, label_fake)
        lossD_fake.backward()
        optimizerD.step()

        acc=evaluation_model.eval(fake_images, one_hot_labels)
        total_acc+=acc

        netG.zero_grad()
        output = netD(fake_images, one_hot_labels).view(-1)
        lossG = criterion(output, label_real)
        loss+=lossG.item()

        lossG.backward()
        optimizerG.step()

    mean_acc = total_acc/(iter)
    print(f"[{epoch}/{num_epochs}], Loss: {loss:.3f}, acc: {mean_acc:.3f}")
    if mean_acc > best:
        best=mean_acc
        if epoch % save_epochs == 0:
            torch.save(netG.state_dict(), os.path.join("./GANcheckpoints", f"netG epoch={epoch} acc={mean_acc:.3f}.pth"))
            torch.save(netD.state_dict(), os.path.join("./GANcheckpoints", f"netD epoch={epoch} acc={mean_acc:.3f}.pth"))
            print(f"save epoch=={epoch} acc={mean_acc:.3f}.pth")
    elif mean_acc < 0.2:
        print(f"total_acc={mean_acc}")
        break
