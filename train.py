import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sresnet
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(100)
torch.cuda.manual_seed(100)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--class_num', default=100, type=int)
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--lambda_KD', default=0.5, type=float)
args = parser.parse_args()
print(args)


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/3.0, dim=1)
    softmax_targets = F.softmax(targets/3.0, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


BATCH_SIZE = 128
LR = 0.1

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset, testset = None, None
if args.class_num == 100:
    print("dataset: CIFAR100")
    trainset = torchvision.datasets.CIFAR100(
        root='/home2/lthpc/data',
        train=True,
        download=False,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root='/home2/lthpc/data',
        train=False,
        download=False,
        transform=transform_test
    )
if args.class_num == 10:
    print("dataset: CIFAR10")
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=transform_test
    )
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

net = None
if args.depth == 18:
    net = sresnet.resnet18(num_classes=args.class_num, align="CONV")
    print("using resnet 18")
if args.depth == 50:
    net = sresnet.resnet50(num_classes=args.class_num, align="CONV")
    print("using resnet 50")
if args.depth == 101:
    net = sresnet.resnet101(num_classes=args.class_num, align="CONV")
    print("using resnet 101")
if args.depth == 152:
    net = sresnet.resnet152(num_classes=args.class_num, align="CONV")
    print("using resnet 152")

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)

if __name__ == "__main__":
    best_acc = 0
    print("Start Training")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(args.epoch):
                correct4, correct3, correct2, correct1, correct0 = 0, 0, 0, 0, 0
                predicted4, predicted3, predicted2, predicted1, predicted0 = 0, 0, 0, 0, 0
                if epoch in [75, 130, 180]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, feature_loss = net(inputs)

                    ensemble = sum(outputs[:-1])/len(outputs)
                    ensemble.detach_()
                    ensemble.requires_grad = False

                    #   compute loss
                    loss = torch.FloatTensor([0.]).to(device)

                    #   for deepest classifier
                    loss += criterion(outputs[0], labels)

                    #   for soft & hard target
                    teacher_output = outputs[0].detach()
                    teacher_output.requires_grad = False

                    for index in range(1, len(outputs)):
                        loss += CrossEntropy(outputs[index], teacher_output) * args.lambda_KD * 9
                        loss += criterion(outputs[index], labels) * (1 - args.lambda_KD)

                    #   for faeture align loss
                    if args.lambda_KD != 0:
                        loss += feature_loss * 5e-7

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total += float(labels.size(0))
                    sum_loss += loss.item()

                    _0, predicted0 = torch.max(outputs[0].data, 1)
                    _1, predicted1 = torch.max(outputs[1].data, 1)
                    _2, predicted2 = torch.max(outputs[2].data, 1)
                    _3, predicted3 = torch.max(outputs[3].data, 1)
                    _4, predicted4 = torch.max(ensemble.data, 1)

                    correct0 += float(predicted0.eq(labels.data).cpu().sum())
                    correct1 += float(predicted1.eq(labels.data).cpu().sum())
                    correct2 += float(predicted2.eq(labels.data).cpu().sum())
                    correct3 += float(predicted3.eq(labels.data).cpu().sum())
                    correct4 += float(predicted4.eq(labels.data).cpu().sum())

                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: 4/4: %.2f%% 3/4: %.2f%% 2/4: %.2f%%  1/4: %.2f%%'
                          ' Ensemble: %.2f%%' % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                                                  100 * correct0 / total, 100 * correct1 / total,
                                                  100 * correct2 / total, 100 * correct3 / total,
                                                  100 * correct4 / total))

                print("Waiting Test!")
                with torch.no_grad():
                    correct4, correct3, correct2, correct1, correct0 = 0, 0, 0, 0, 0
                    predicted4, predicted3, predicted2, predicted1, predicted0 = 0, 0, 0, 0, 0
                    correct = 0.0
                    total = 0.0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs, feature_loss = net(images)
                        ensemble = sum(outputs) / len(outputs)
                        _0, predicted0 = torch.max(outputs[0].data, 1)
                        _1, predicted1 = torch.max(outputs[1].data, 1)
                        _2, predicted2 = torch.max(outputs[2].data, 1)
                        _3, predicted3 = torch.max(outputs[3].data, 1)
                        _4, predicted4 = torch.max(ensemble.data, 1)

                        correct0 += float(predicted0.eq(labels.data).cpu().sum())
                        correct1 += float(predicted1.eq(labels.data).cpu().sum())
                        correct2 += float(predicted2.eq(labels.data).cpu().sum())
                        correct3 += float(predicted3.eq(labels.data).cpu().sum())
                        correct4 += float(predicted4.eq(labels.data).cpu().sum())
                        total += float(labels.size(0))

                    print('Test Set AccuracyAcc: 4/4: %.4f%% 3/4: %.4f%% 2/4: %.4f%%  1/4: %.4f%%'
                          ' Ensemble: %.4f%%' % (100 * correct0 / total, 100 * correct1 / total,
                                                 100 * correct2 / total, 100 * correct3 / total,
                                                 100 * correct4 / total))
                    if correct0/total > best_acc:
                        torch.save(net.state_dict(), "./4att/bestmodel.pth")
                        print("model saved")
                        best_acc = correct0/total

            print("Training Finished, TotalEPOCH=%d" % args.epoch)



