import net
import torch.optim as optim
import torch.nn as nn
import torch
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netI = net.NetIdentifierResNet34()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    netI = nn.DataParallel(netI)

netI.to(device)
netI.train()

img_s = torch.zeros(40, 3, 128, 128, device=device)
label_s = torch.zeros(40, dtype=torch.long, device=device)

loss_id = nn.CrossEntropyLoss()
optimizer_id = optim.Adam(netI.parameters())

start = time.time()
for i in range(200):
    optimizer_id.zero_grad()
    cls_prob_score = netI(img_s)[0]
    loss = loss_id(cls_prob_score, label_s)
    loss.backward()
    optimizer_id.step()

print(time.time()-start)