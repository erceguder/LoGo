import math
from tqdm import tqdm

# lr scheduler for training
def decay_learning_rate(optimizer, epoch, args):
    """
        lr = min_lr + 0.5*(max_lr - min_lr) * (1 + cos(pi * t/T))
    """
    lr = args.min_lr + 0.5*(args.max_lr - args.min_lr) * (1. + math.cos(math.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# train for one epoch
def train(net, data_loader, optimizer, epoch, args):
    net.train()
    decay_learning_rate(optimizer, epoch, args)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    for global_crops, local_crops in train_bar:
        g_1, g_2 = global_crops[0].to(args.device), global_crops[1].to(args.device)
        l_1, l_2 = local_crops[0].to(args.device), local_crops[1].to(args.device)
#        im_1, im_2 = im_1.to(args.device), im_2.to(args.device)
        continue

        loss = net(im_1, im_2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size

        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num
