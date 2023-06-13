import parse
import model
import cifar
import train
import test
import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    args = parse.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # data prepare
    train_data = cifar.CIFAR10Pair(root='data', train=True, download=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # create models
    q_encoder = model.ModelBase(
        feature_dim=args.moco_dim,
        arch=args.arch,
        bn_splits=args.bn_splits
    ).to(args.device)

    k_encoder = model.ModelBase(
        feature_dim=args.moco_dim,
        arch=args.arch,
        bn_splits=args.bn_splits
    ).to(args.device)

    for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient

    mlp = model.MLP().to(args.device)

    # define optimizer
    q_optimizer = torch.optim.SGD(q_encoder.parameters(), lr=args.max_lr, weight_decay=args.wd, momentum=0.9)
    mlp_optimizer = torch.optim.SGD(mlp.parameters(), lr=args.max_lr, weight_decay=args.wd, momentum=0.9)

    # logging
    results = {'train_loss': [], 'test_acc@1': []}

    # training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train.train(q_encoder, k_encoder, mlp, train_loader, q_optimizer, mlp_optimizer, epoch, args)
        results['train_loss'].append(train_loss)

#        test_acc_1 = test.test(model.encoder_q, memory_loader, test_loader, epoch, args)
#        results['test_acc@1'].append(test_acc_1)

    print(results)