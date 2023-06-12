import math
from tqdm import tqdm

# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# train for one epoch
def train(net, data_loader, optimizer, epoch, args):
    net.train()
    adjust_learning_rate(optimizer, epoch, args)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    for im_1, im_2 in train_bar:
        im_1, im_2 = im_1.to(args.device), im_2.to(args.device)

        loss = net(im_1, im_2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size

        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num
