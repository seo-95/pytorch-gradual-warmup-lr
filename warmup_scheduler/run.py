import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.sgd import SGD

from warmup_scheduler import GradualWarmupScheduler

if __name__ == '__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optim = SGD(model, 0.1)

    # scheduler_warmup is chained with schduler_steplr
    scheduler_steplr = MultiStepLR(optim, milestones=[300, 400], gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, warmup_steps=200, post_warmup_scheduler=scheduler_steplr)

    lr_values = []
    global_step = 0
    for epoch in range(1, 10):
        for step in range(1, 50):
            global_step += 1
            lr_values.append(optim.param_groups[0]['lr'])
            optim.step()    # backward pass (update network)
            scheduler_warmup.step()
    plt.plot(np.arange(1, len(lr_values)+1), lr_values, color='blue')
    plt.title('Learning rate trend')
    plt.xlabel('Steps')
    plt.savefig('lr.png')
    print('result saved in lr.png')
