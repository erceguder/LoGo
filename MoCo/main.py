import parse
import model
import cifar
import train
import test
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    args = parse.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]
    )
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]
    )
    # data prepare
    train_data = cifar.CIFAR10Pair(root='data', train=True, transform=train_transform, download=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    memory_data = cifar.CIFAR10(root='data', train=True, transform=test_transform, download=False)
    memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_data = cifar.CIFAR10(root='data', train=False, transform=test_transform, download=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # create model
    model = model.ModelMoCo(
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        arch=args.arch,
        bn_splits=args.bn_splits,
        symmetric=args.symmetric,
        device=args.device
    ).to(args.device)

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.max_lr, weight_decay=args.wd, momentum=0.9)

    # logging
    results = {'train_loss': [], 'test_acc@1': []}

    # training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train.train(model, train_loader, optimizer, epoch, args)
        results['train_loss'].append(train_loss)

#        test_acc_1 = test.test(model.encoder_q, memory_loader, test_loader, epoch, args)
#        results['test_acc@1'].append(test_acc_1)

    print(results)