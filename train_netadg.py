import net
import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import grad
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netI = net.NetIdentifierResNet34()
netA = net.NetAttributeResNet34()
netG = net.NetGenerator()
netD = net.NetDiscriminator()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    netI = nn.DataParallel(netI)
    netA = nn.DataParallel(netA)
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

netI = netI.to(device)
netA = netA.to(device)
netG = netG.to(device)
netD = netD.to(device)

netI.eval()
netA.train()
netG.train()
netD.train()

for x in netI.parameters():
    x.requires_grad_(False)

img_s = torch.zeros(20, 3, 128, 128, device=device)
label_s = torch.zeros(20, dtype=torch.long, device=device)
img_a = torch.zeros(20, 3, 128, 128, device=device)
label_a = torch.zeros(20, dtype=torch.long, device=device)

optimizer_netA = optim.Adam(netA.parameters())
optimizer_netG = optim.Adam(netG.parameters())
optimizer_netD = optim.Adam(netD.parameters())


def loss_gp(img_g, img_a):
    t = torch.rand(1, dtype=img_g.dtype, layout=img_g.layout, device=img_g.device)
    img_t = img_g * t + img_a * (1 - t)
    img_t.requires_grad_()
    d_score_t = netD(img_t)[0]
    gradients = grad(outputs=d_score_t, inputs=img_t,
                     grad_outputs=torch.ones_like(d_score_t),
                     create_graph=True)[0]
    gp = ((gradients.norm(p=2, dim=1) - 1) ** 2).mean()
    return gp


for i in range(20):
    start = time.time()
    optimizer_netA.zero_grad()
    optimizer_netG.zero_grad()
    cls_prob_s, id_s, feature_gc_s = netI(img_s)
    id_s_data = id_s.detach()
    feature_gc_s_data = feature_gc_s.detach()

    mean, log_variance = netA(img_a)
    variance = torch.exp(log_variance)
    loss_gaussian = torch.sum(mean ** 2) + torch.sum(variance - log_variance - 1)

    attr_a = mean + torch.randn_like(log_variance) * torch.sqrt(variance)
    img_g = netG(id_s_data, attr_a)
    loss_reconstruction = torch.sum((img_a - img_g) ** 2)

    loss_netA = loss_gaussian + loss_reconstruction
    loss_netA.backward()

    attr_a = attr_a.detach()
    img_g = netG(id_s_data, attr_a)
    cls_prob_a, id_g, feature_gc_g = netI(img_g)
    loss_gc = torch.sum((feature_gc_g - feature_gc_s_data) ** 2)

    feature_gd_g = netD(img_g)[1]
    feature_gd_a = netD(img_a)[1]
    loss_gd = torch.sum((feature_gd_g - feature_gd_a.detach()) ** 2)
    loss_netG = loss_gd + loss_gc  # loss_reconstruction is in loss_netA
    loss_netG.backward()

    img_g = img_g.detach()
    optimizer_netD.zero_grad()
    d_score_g = netD(img_g)[0]
    d_score_a = netD(img_a)[0]
    loss_netD = torch.mean(d_score_g) - torch.mean(d_score_a) + 10 * loss_gp(img_g, img_a)
    loss_netD.backward()
    optimizer_netA.step()
    optimizer_netG.step()
    optimizer_netD.step()
    print(time.time()-start)
