import math
import torch
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
def train(q_encoder, k_encoder, mlp, data_loader, q_optimizer, mlp_optimizer, epoch, args):
    q_encoder.train()
    mlp.train()

    decay_learning_rate(q_optimizer, epoch, args)
    decay_learning_rate(mlp_optimizer, epoch, args)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    ce_criterion = torch.nn.CrossEntropyLoss()

    for global_crops, local_crops in train_bar:
        N = global_crops[0].shape[0]

        # N x 3 x 32 x 32
        g_1, g_2 = global_crops[0].to(args.device), global_crops[1].to(args.device)
        l_1, l_2 = local_crops[0].to(args.device), local_crops[1].to(args.device)

        # N x 128
        z_g_1, z_g_2 = q_encoder(g_1), q_encoder(g_2)
        z_l_1, z_l_2 = q_encoder(l_1), q_encoder(l_2)

        k = torch.randperm(n=N) # hope no k=j

        omega = mlp.omega_loss(z_l_1, z_l_2, z_l_1[k])

        omega.backward(retain_graph=True)
        mlp_optimizer.step()
        mlp_optimizer.zero_grad()

        # no shuffling needed on query encoder
        # N x N, positives on diagonals
        # NOT symmetric
        logits = torch.mm(z_g_1, z_g_2.t())
        logits /= args.moco_t

        labels = torch.arange(N, device=args.device)

        #  mean over the batch
        L_gg = ce_criterion(logits, labels)

        with torch.no_grad():
            # N x 128
            m_z_g_1 = k_encoder(g_1)
            m_z_g_2 = k_encoder(g_2)

        #  mean over the batch
        logits = torch.mm(z_l_1, m_z_g_1.t())
        L_lg = ce_criterion(logits, labels)

        logits = torch.mm(z_l_1, m_z_g_2.t())
        L_lg += ce_criterion(logits, labels)

        logits = torch.mm(z_l_2, m_z_g_1.t())
        L_lg += ce_criterion(logits, labels)

        logits = torch.mm(z_l_2, m_z_g_2.t())
        L_lg += ce_criterion(logits, labels)

        #  mean over the batch
        L_ll = mlp(z_l_1, z_l_2).mean()

        loss = L_gg + L_lg + args.lam * L_ll

        loss.backward()
        q_optimizer.step()
        q_optimizer.zero_grad()

        total_num += N
        total_loss += loss * N

        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, q_optimizer.param_groups[0]['lr'], total_loss / total_num))

        # momentum update
        for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    return total_loss / total_num