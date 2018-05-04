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

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            kwargs['global_step'] = self.count
            return getattr(self.writer, name)(*args, **kwargs)

        return wrapper
