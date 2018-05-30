import os

import torch
from tensorboardX import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MultiStepStatisticCollector:
    def __init__(self):
        self.count = 0
        self.writer = SummaryWriter()

    def close(self):
        self.writer.close()

    def next_step(self):
        self.count = self.count + 1

    def current_step(self):
        return self.count

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            kwargs['global_step'] = self.count
            return getattr(self.writer, name)(*args, **kwargs)

        return wrapper


def save_nets(nets, info, folder):
    """
    :param nets:  {'net_name':(model,optimizer)}
    :param info:  any other stats like epoch,loss,...
    :param folder:
    :return:
    """
    os.makedirs(folder, exist_ok=True)
    for net_name, v in nets.items():
        if 'optimizer' in v:
            model = v['model']
            optimizer = v['optimizer']
            path = os.path.join(folder, net_name)
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)
    torch.save(info, os.path.join(folder, "info"))
    print("saved checkpoint to directory {}".format(os.path.abspath(folder)))


def load_nets(nets, folder):
    for net_name, v in nets.items():
        model = v['model']
        path = os.path.join(folder, net_name)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in v:
            v['optimizer'].load_state_dict(checkpoint['optimizer'])
    return torch.load(os.path.join(folder, "info"))


class MySampler:
    def __init__(self, dataset, batch_size, num_workers):
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers, pin_memory=True)
        self.batch_size = batch_size
        self.iterator = iter(self.loader)

    def next(self):
        try:
            b = next(self.iterator)
            size = b[0].size(0)
            if size != self.batch_size:
                self.iterator = iter(self.loader)
                b = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            b = next(self.iterator)
        return b

    def len_samples(self):
        return len(self.loader)
