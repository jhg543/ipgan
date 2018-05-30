import argparse
import os
import time
from contextlib import closing

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import net as networks
import utils
from utils import AverageMeter
from utils import MultiStepStatisticCollector
from utils import load_nets
from utils import save_nets

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=40, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save checkpoints (default: .)')
parser.add_argument('--neti', default='neti.tar', type=str, metavar='PATH',
                    help='path to id model (default: neti.tar)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
args = parser.parse_args()


def main():
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    # TODO what is pin_memory?
    train_sampler = utils.MySampler(train_dataset, args.batch_size, args.workers)

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

    val_sampler = utils.MySampler(val_dataset, args.batch_size, args.workers)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    net_i = networks.NetIdentifierResNet34(num_classes=len(train_dataset.classes))
    net_a = networks.NetAttributeResNet34()
    net_g = networks.NetGenerator()
    net_d = networks.NetDiscriminator()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net_i = nn.DataParallel(net_i)
        net_a = nn.DataParallel(net_a)
        net_g = nn.DataParallel(net_g)
        net_d = nn.DataParallel(net_d)

    net_i = net_i.to(device)
    net_a = net_a.to(device)
    net_g = net_g.to(device)
    net_d = net_d.to(device)

    net_i.eval()

    for x in net_i.parameters():
        x.requires_grad_(False)

    optimizer_net_a = optim.Adam(net_a.parameters())
    optimizer_net_g = optim.Adam(net_g.parameters())
    optimizer_net_d = optim.Adam(net_d.parameters())

    nets = {'netA': {'model': net_a, 'optimizer': optimizer_net_a},
            'netG': {'model': net_g, 'optimizer': optimizer_net_g},
            'netD': {'model': net_d, 'optimizer': optimizer_net_d},
            'netI': {'model': net_i}}

    model_neti = torch.load(args.neti)
    net_i.load_state_dict(model_neti['state_dict'])
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.exists(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = load_nets(nets, args.resume)
            args.start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    with closing(MultiStepStatisticCollector()) as stat_log:
        for epoch in range(args.start_epoch, args.epochs):
            train(device, train_sampler, nets, stat_log)
            validate(device, val_sampler, nets, stat_log)
            checkpoint_path = os.path.join(args.save, "checkpoint", "epoch" + str(epoch))
            save_nets(nets, {'epoch': epoch}, checkpoint_path)
            print("checkpoint saved to {}".format(os.path.abspath(checkpoint_path)))


def train(device, train_sampler, nets, stat_log):
    net_i, net_g, net_d, net_a = [nets[net_name]['model'] for net_name in ['netI', 'netG', 'netD', 'netA']]
    net_i.eval()
    net_g.train()
    net_d.train()
    net_a.train()
    for i in range(train_sampler.len_samples() // 3):
        log_img = (i == 0)
        metric_rec, img_rec = train_onebatch(device, train_sampler, nets, False, log_img)
        metric_gen, img_gen = train_onebatch(device, train_sampler, nets, True, log_img)

        for marker, metric in [('rec', metric_rec), ('gen', metric_gen)]:
            loss_rec, loss_gd, loss_gc, loss_net_d, loss_n01, batch_time = metric
            if i % args.print_freq == 0:
                print(("batch={} type={} loss_rec={} "
                       "loss_gd={} loss_gc={} "
                       "loss_critic={} loss_n01={} "
                       "batch_time={}").format(i, marker, *metric))
            stat_log.add_scalars('train/loss_a_' + marker,
                                 {'loss_rec': loss_rec, 'loss_n': loss_n01})
            stat_log.add_scalars('train/loss_g_' + marker,
                                 {'loss_rec': loss_rec, 'loss_gd': loss_gd,
                                  'loss_gc': loss_gc})
            stat_log.add_scalars('train/loss_d_' + marker,
                                 {'loss_net_d': loss_net_d})
            stat_log.add_scalar('train/batchtime_' + marker, batch_time)

        if log_img:
            stat_log.add_image("train/img_rec", img_rec)
            stat_log.add_image("train/img_gen", img_gen)
            log_net_histogram(nets, stat_log)
        stat_log.next_step()


def loss_d(score_g, score_a):
    return torch.mean(nn.functional.relu(1.0 + score_g)) + torch.mean(
        nn.functional.relu(1.0 - score_a))


def train_onebatch(device, train_sampler, nets, is_gen, log_img=False):
    net_i, net_g, net_d, net_a = [nets[net_name]['model'] for net_name in ['netI', 'netG', 'netD', 'netA']]
    optimizer_g, optimizer_d, optimizer_a = [nets[net_name]['optimizer'] for net_name in ['netG', 'netD', 'netA']]

    start_batch_time = time.time()
    img_s, _ = train_sampler.next()
    weight_reconstruction = 1.0
    img_a = img_s
    if is_gen:
        weight_reconstruction = 0.1
        img_a, _ = train_sampler.next()
    img_s = img_s.cuda(device)
    img_a = img_a.cuda(device)
    batch_size = img_s.size(0)

    optimizer_a.zero_grad()
    optimizer_g.zero_grad()
    cls_prob_s, id_s, feature_gc_s = net_i(img_s)
    id_s_data = id_s.detach()
    feature_gc_s_data = feature_gc_s.detach()

    mean, log_variance = net_a(img_a)
    variance = torch.exp(log_variance)
    loss_n01 = (torch.sum(mean ** 2) + torch.sum(variance - log_variance - 1)) / batch_size

    latent_attr_a = mean + torch.randn_like(log_variance) * torch.sqrt(variance)
    img_g = net_g(id_s_data, latent_attr_a)
    loss_rec = torch.sum((img_a - img_g) ** 2) / batch_size * weight_reconstruction

    loss_net_a = loss_n01 + loss_rec
    loss_net_a.backward()

    latent_attr_a = latent_attr_a.detach()
    img_g = net_g(id_s_data, latent_attr_a)
    cls_prob_a, id_g, feature_gc_g = net_i(img_g)
    loss_gc = torch.sum((feature_gc_g - feature_gc_s_data) ** 2) / batch_size

    # feature_gd_g = net_d(img_g)[1]
    # feature_gd_a = net_d(img_a)[1]
    # loss_gd = torch.norm(feature_gd_g - feature_gd_a.detach(), 2)
    loss_gd = -torch.mean(net_d(img_g)[0])
    loss_net_g = loss_gd + loss_gc  # loss_reconstruction gradient already bp-ed in loss_net_a
    loss_net_g.backward()
    optimizer_a.step()
    optimizer_g.step()

    mean, log_variance = net_a(img_a)
    variance = torch.exp(log_variance)
    latent_attr_a = mean + torch.randn_like(log_variance) * torch.sqrt(variance)
    img_g = net_g(id_s_data, latent_attr_a)
    img_g = img_g.detach()
    optimizer_d.zero_grad()
    d_score_g = net_d(img_g)[0]
    d_score_a = net_d(img_a)[0]
    loss_critic = loss_d(d_score_g, d_score_a)
    loss_net_d = loss_critic
    loss_net_d.backward()
    optimizer_d.step()

    batch_time = time.time() - start_batch_time
    losses = [loss_rec, loss_gd, loss_gc, loss_net_d, loss_n01]
    metrics = [loss.item() for loss in losses]
    metrics.append(batch_time)
    img = None
    if log_img:
        img = make_grid_3(img_s, img_a, img_g)
    return metrics, img


def validate_onebatch(device, val_sampler, nets, is_gen, log_img=False):
    """

    :param val_sampler:
    :param nets:
    :param is_gen:
    :return: (loss_rec, loss_gd, loss_gc, loss_critic, batch_time)
    """
    net_i, net_g, net_d, net_a = [nets[net_name]['model'] for net_name in ['netI', 'netG', 'netD', 'netA']]
    start_batch_time = time.time()
    img_s, _ = val_sampler.next()
    weight_rec = 1.0
    img_a = img_s
    if is_gen == 1:
        weight_rec = 0.1
        img_a, _ = val_sampler.next()
    img_s = img_s.cuda(device)
    img_a = img_a.cuda(device)
    _, id_s, feature_gc_s = net_i(img_s)

    mean, _ = net_a(img_a)
    latent_attr_a = mean
    img_g = net_g(id_s, latent_attr_a)
    loss_rec = torch.norm(img_a - img_g, 2) * weight_rec

    _, _, feature_gc_g = net_i(img_g)
    loss_gc = torch.norm(feature_gc_g - feature_gc_s, 2)

    # feature_gd_g = net_d(img_g)[1]
    # feature_gd_a = net_d(img_a)[1]
    # loss_gd = torch.norm(feature_gd_g - feature_gd_a,2)
    loss_gd = torch.tensor([0], device=device)

    d_score_g = net_d(img_g)[0]
    d_score_a = net_d(img_a)[0]
    loss_critic = torch.mean(nn.functional.relu(1.0 + d_score_g)) + torch.mean(nn.functional.relu(1.0 - d_score_a))
    batch_time = time.time() - start_batch_time
    losses = [loss_rec, loss_gd, loss_gc, loss_critic]
    metrics = [loss.item() for loss in losses]
    metrics.append(batch_time)
    img = None
    if log_img:
        img = make_grid_3(img_s, img_a, img_g)
    return metrics, img


def validate(device, val_sampler, nets, stat_log):
    net_i, net_g, net_d, net_a = [nets[net_name]['model'] for net_name in ['netI', 'netG', 'netD', 'netA']]
    net_i.eval()
    net_g.eval()
    net_d.eval()
    net_a.eval()

    meters_rec = [AverageMeter() for i in range(5)]
    meters_gen = [AverageMeter() for i in range(5)]

    with torch.no_grad():
        for i in range(val_sampler.len_samples() // 3):
            log_img = (i == 0)
            metrics_rec, img_rec = validate_onebatch(device, val_sampler, nets, False, log_img)
            for meter, metric in zip(meters_rec, metrics_rec):
                meter.update(metric)

            metrics_gen, img_gen = validate_onebatch(device, val_sampler, nets, True, log_img)
            for meter, metric in zip(meters_gen, metrics_gen):
                meter.update(metric)
            if log_img:
                stat_log.add_image("val/img_rec", img_rec)
                stat_log.add_image("val/img_gen", img_gen)

        for marker, meters in [('rec', meters_rec), ('gen', meters_gen)]:
            loss_rec, loss_gd, loss_gc, loss_critic, batch_time = [i.avg for i in meters]
            print(("VALIDATION type={} loss_rec={} "
                   "loss_gd={} loss_gc={} "
                   "loss_critic={} batch_time={}").format(marker, *[i.avg for i in meters]))
            stat_log.add_scalars('val/loss_g_' + marker,
                                 {'loss_rec': loss_rec, 'loss_gd': loss_gd,
                                  'loss_gc': loss_gc})
            stat_log.add_scalar('val/loss_critic_' + marker, loss_critic)
            stat_log.add_scalar('val/batch_time_' + marker, batch_time)
        stat_log.next_step()


def make_grid_3(img_s, img_a, img_g):
    imgs = torch.stack([img_s, img_a, img_g])
    imgs = torch.transpose(imgs, 0, 1)
    imgs = torch.reshape(imgs, (-1, *imgs.size()[2:]))
    imgs = make_grid(imgs, 9)
    for t, m, s in zip(imgs, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    return imgs


def log_net_histogram(nets, stat_log):
    for net_name, v in nets.items():
        net = v['model']
        for name, param in net.named_parameters():
            stat_log.add_histogram("{}-{}".format(net_name, name), param)


if __name__ == "__main__":
    main()
