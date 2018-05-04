import time
import argparse
import os

import net
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import AverageMeter
from utils import MultiStepStatisticCollector
import shutil
from contextlib import closing

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
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
args = parser.parse_args()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_i = net.NetIdentifierResNet34()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net_i = nn.DataParallel(net_i)

    net_i.to(device)

    loss_i = nn.CrossEntropyLoss()
    optimizer_i = optim.Adam(net_i.parameters())

    best_prec1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net_i.load_state_dict(checkpoint['state_dict'])
            optimizer_i.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

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

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, net_i, device, loss_i)
        return

    with closing(MultiStepStatisticCollector()) as stat_log:
        for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            train(train_loader, net_i, optimizer_i, device, loss_i, epoch, stat_log)

            # evaluate on validation set
            prec1 = validate(val_loader, net_i, device, loss_i, stat_log)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net_i.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer_i.state_dict(),
            }, is_best)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, net_i, optimizer, device, loss_module, epoch, stat_log=None):
    net_i.train()

    end = time.time()
    for i, (img_s, label_s) in enumerate(train_loader):
        batch_size = img_s.size(0)
        label_s = label_s.cuda(device=device, non_blocking=True)

        # compute output
        cls_prob_score = net_i(img_s)[0]
        loss = loss_module(cls_prob_score, label_s)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(cls_prob_score, label_s, topk=(1, 5))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()

        # record loss
        loss_data = loss.item()
        top1_data = prec1[0]
        top5_data = prec5[0]

        if stat_log is not None:
            stat_log.add_scalar('train/loss', loss_data)
            stat_log.add_scalars('train/prec', {'top1': top1_data, 'top5': top5_data})
            stat_log.add_scalar('train/batchtime', batch_time)
            stat_log.next_step()

        # print
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time:.3f}\t'
                  'Loss {loss:.4f}\t'
                  'Prec@1 {top1:.3f}\t'
                  'Prec@5 {top5:.3f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=loss_data, top1=top1_data, top5=top5_data))


def validate(val_loader, netI, device, loss_module, stat_log=None):
    # switch to evaluate mode
    netI.eval()

    # average on all data
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for i, (img_s, label_s) in enumerate(val_loader):
            batch_size = img_s.size(0)
            label_s = label_s.cuda(device=device, non_blocking=True)

            # compute output
            cls_prob_score = netI(img_s)[0]
            loss = loss_module(cls_prob_score, label_s)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(cls_prob_score, label_s, topk=(1, 5))

            # avg
            loss_data = loss.item()
            top1_data = prec1[0]
            top5_data = prec5[0]

            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss_data, batch_size)
            top1.update(top1_data, batch_size)
            top5.update(top5_data, batch_size)

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        if stat_log is not None:
            stat_log.add_scalars('validation/prec', {'top1': top1.avg, 'top5': top5.avg})
            stat_log.add_scalar('validation/loss', losses.avg)
            stat_log.next_step()

    return top1.avg


if __name__ == "__main__":
    main()

# img_s = torch.zeros(40, 3, 128, 128, device=device)
# label_s = torch.zeros(40, dtype=torch.long, device=device)
