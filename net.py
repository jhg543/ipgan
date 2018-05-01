import torchvision.models as models
import torch
import summary
import torch.nn as nn
import math


class NetIdentifierResNet34(nn.Module):

    def __init__(self, num_classes=1000):
        layers = [3, 4, 6, 3]
        block = models.resnet.BasicBlock
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        id_feature = x

        x = self.avgpool(x)
        pool_feature = x
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, id_feature, pool_feature


class NetAttributeResNet34(nn.Module):

    def __init__(self, feature_dim=1024):
        layers = [3, 4, 6, 3]
        block = models.resnet.BasicBlock
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, feature_dim)
        self.fc_mean = nn.Linear(feature_dim, feature_dim)
        self.fc_log_var = nn.Linear(feature_dim, feature_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        mean = self.fc_mean(x)
        log_variance = self.fc_log_var(x)

        return mean, log_variance


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt((torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8))

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)


netG_act = nn.LeakyReLU
netG_pn = PixelNorm


class NetGenerator(nn.Module):

    def __init__(self, id_size=4, id_channels=512, attr_size=1024):
        super().__init__()
        self.act = netG_act()
        self.pn = netG_pn()
        self.pn2 = netG_pn()
        self.fc_attr = nn.Linear(attr_size, id_channels * id_size * id_size)
        self.after_concat = nn.Sequential(*[
            netG_pn(),
            nn.ConvTranspose2d(id_channels * 2, 512, 3, padding=1),
            netG_act(),
            netG_pn(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            netG_act(),
            netG_pn()
        ])
        self.layer1 = self._make_layer(3, 512, 512)
        self.layer2 = self._make_layer(3, 512, 256)
        self.layer3 = self._make_layer(3, 256, 128)
        self.layer4 = self._make_layer(3, 128, 96)
        self.layer5 = self._make_layer(3, 96, 64)
        self.torgb = nn.Conv2d(64, 3, 5, padding=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, conv_kernel_size, in_channels, out_channels):
        layers = [
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels, out_channels, conv_kernel_size, padding=(conv_kernel_size - 1) // 2),
            netG_act(),
            netG_pn(),
            nn.Conv2d(out_channels, out_channels, 1),
            netG_act(),
            netG_pn()
        ]
        return nn.Sequential(*layers)

    def forward(self, id_feature, attr_feature):
        x = self.fc_attr(attr_feature)
        x = x.view(*id_feature.size())
        x = self.act(x)
        x = self.pn(x)
        x = torch.cat([self.pn2(id_feature), x], dim=1)
        x = self.after_concat(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.torgb(x)

        return x


netD_act = nn.LeakyReLU


class NetDiscriminator(nn.Module):

    def __init__(self, last_feature_size=4):
        super().__init__()
        self.fromrgb = nn.Sequential(*[
            nn.Conv2d(3, 64, 1),
            netD_act()
        ])
        self.layer1 = self._make_layer(3, 64, 96)
        self.layer2 = self._make_layer(3, 96, 128)
        self.layer3 = self._make_layer(3, 128, 256)
        self.layer4 = self._make_layer(3, 256, 512)
        self.layer5 = self._make_layer(3, 512, 512)
        self.last_feature_conv = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1 = nn.Linear(512 * last_feature_size * last_feature_size, 512)
        self.fc2 = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, conv_kernel_size, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, in_channels, conv_kernel_size, padding=(conv_kernel_size - 1) // 2),
            netD_act(),
            nn.Conv2d(in_channels, out_channels, conv_kernel_size, padding=(conv_kernel_size - 1) // 2),
            netD_act(),
            nn.AvgPool2d(2, 2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fromrgb(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.last_feature_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        feature_gd = x
        x = self.fc2(x)
        return x, feature_gd


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("+++++NetG+++++")
    m = NetGenerator(id_size=4)
    m = m.to(device)
    print(m)
    m(torch.empty(2,512,4,4,device=device),torch.empty(2,1024,device=device))
    #summary.summary(m, [(512, 4, 4), (1024,)])
    print("+++++NetD+++++")
    m = NetDiscriminator()
    m = m.to(device)
    print(m)
    m(torch.empty(2, 3, 128, 128, device=device))
    #summary.summary(m, (3, 128, 128))

