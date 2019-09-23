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


def judge(tensor, c):
    dic = {0: 0.98, 1: 0.97, 2: 0.98, 3: 0.95}
    maxium = torch.max(tensor)
    if float(maxium) > dic[c]:
        return True
    else:

        return False


BATCH_SIZE = 256
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
net.load_state_dict(torch.load("bestmodel.pth"))


if __name__ == "__main__":
    best_acc = 0
    caught = [0, 0, 0, 0, 0]
    print("Waiting Test!")
    with torch.no_grad():
        correct4, correct3, correct2, correct1, correct0 = 0, 0, 0, 0, 0
        predicted4, predicted3, predicted2, predicted1, predicted0 = 0, 0, 0, 0, 0
        correct = 0.0
        total = 0.0
        right = 0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, feature_loss = net(images)
            ensemble = sum(outputs) / len(outputs)
            outputs.reverse()

            for index in range(len(outputs)):
                outputs[index] = F.softmax(outputs[index])

            for index in range(images.size(0)):
                ok = False
                for c in range(4):
                    logits = outputs[c][index]
                    if judge(logits, c):
                        caught[c] += 1
                        predict = torch.argmax(logits)
                        if predict.cpu().numpy().item() == labels[index]:
                            right += 1

                        ok = True
                        break

                if not ok:
                    caught[-1] += 1
                    #   print(index, "ensemble")
                    logits = ensemble[index]
                    predict = torch.argmax(logits)
                    if predict.cpu().numpy().item() == labels[index]:
                        right += 1

            total += float(labels.size(0))
        print('Test Set Accuracy:  %.4f%% ' % (100 * right / total))
        acceleration_ratio = 1/((0.32 * caught[0] + 0.53* caught[1] + 0.76*caught[2] + 1.0 * caught[3] + 1.07 * caught[4])/total)

        print("Acceleration ratio:", acceleration_ratio)



