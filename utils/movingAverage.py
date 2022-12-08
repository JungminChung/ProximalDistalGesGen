class MovingAverage(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add_item(self, val, n=1):
        self.val = val
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count