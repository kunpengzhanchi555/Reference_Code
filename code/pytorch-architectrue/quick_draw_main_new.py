import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from functools import partial
import argparse
import os
import time
import shutil
from quick_draw_dataset import quick_draw_dataset
from utils import progress_bar

train_root = "/root/hdd/yankun/Kaggle/quick_draw/all"
train_source = "meta_file"
val_root = "/root/hdd/yankun/Kaggle/quick_draw/all"
val_source = "meta_file_val"
log_source = "log_file"
log_path = os.path.join(train_root, log_source)
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description="quick_draw_training")
# parser.add_argument('--train-root', default='/mnt/lustre/yangkunlin/furniture/data/', type=str)
# parser.add_argument('--train-source', default='/mnt/lustre/yangkunlin/furniture/data/Pseudo-Label3.txt', type=str)
# parser.add_argument('--val-root', default='/mnt/lustre/yangkunlin/furniture/data/val/', type=str)
# parser.add_argument('--val-source', default='/mnt/lustre/yangkunlin/furniture/data/valid.txt', type=str)
# parser.add_argument('--save-path', default='checkpoint6', type=str)
parser.add_argument('--print-freq', '-p', default=10, type=int)
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--epochs', default=20, type=int)
# parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('-b', '--batch-size', default=64, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument("--resume", '-r', action="store_true",
                    help="resume from checkpoint")
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0
end_epoch = 20

patience = 0
min_loss = 100000.0

CLASS_NAME =\
    ['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'airplane', 'alarm_clock', 'ambulance', 'angel',
     'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn',
     'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee',
     'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book',
     'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom',
     'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel',
     'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan',
     'cell_phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup',
     'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship',
     'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser',
     'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses',
     'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo',
     'flashlight', 'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan',
     'garden', 'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger',
     'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck',
     'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant',
     'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'ladder', 'lantern', 'laptop',
     'leaf', 'leg', 'light_bulb', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox',
     'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito',
     'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean',
     'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint_can', 'paintbrush', 'palm_tree', 'panda', 'pants',
     'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano',
     'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond',
     'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain',
     'rainbow', 'rake', 'remote_control', 'rhinoceros', 'river', 'roller_coaster', 'rollerskates', 'sailboat',
     'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw',
     'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag',
     'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat',
     'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo',
     'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine',
     'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 't-shirt', 'table', 'teapot', 'teddy-bear',
     'telephone', 'television', 'tennis_racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth',
     'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle',
     'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine',
     'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch',
     'yoga', 'zebra', 'zigzag']


class FineTuneModel(nn.Module):

    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(
                *list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(original_model.fc.in_features, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg16'
        elif arch.startswith('dense'):
            self.features = nn.Sequential(
                *list(original_model.children())[:-1])

            # Get number of features of last layer
            num_feats = original_model.classifier.in_features

            # Plug our classifier
            self.classifier = nn.Sequential(
                nn.Linear(num_feats, num_classes)
            )
            self.modelName = 'densenet'
        else:
            raise ("Finetuning not supported on this architecture yet")

        for m in self.classifier:
            nn.init.kaiming_normal(m.weight)

        # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'densenet':
            f = F.relu(f, inplace=True)
            f = F.avg_pool2d(f, kernel_size=7).view(f.size(0), -1)
        y = self.classifier(f)
        return y


def main():
    global start_epoch
    global end_epoch
    global patience
    global min_loss
    end_epoch = 20
    args = parser.parse_args()
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('inception_v3'):
        print('inception_v3 without aux_logits!')
        image_size = 341
        input_size = 299
        net = models.__dict__[args.arch](pretrained=True)
    else:
        image_size = 256
        input_size = 224
        net = models.__dict__[args.arch](pretrained=True)
    net = FineTuneModel(net, args.arch, 340)

    net = net.to(device)
    # to choose cpu or gpu
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    print("==> create model is done!")
    # net = net.cuda() # this is the old pytorch version, use the new version—net.to(device)
    # print("args.batch_size{}".format(args.batch_size))
    # Data loading code
    # dataset
    print("==> Preparing data...")
    image_size = 256
    input_size = 224
    normalize = transforms.Normalize(mean=[124.0 / 255.0, 117.0 / 255.0, 104.0 / 255.0],
                                     std=[1.0 / (.0167 * 255)] * 3)
    train_dataset = quick_draw_dataset(
        train_root,
        train_source,
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # ColorAugmentation(),
            normalize,
        ]))
    val_dataset = quick_draw_dataset(
        val_root,
        val_source,
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(input_size),
            # transforms.Resize(input_size),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]))

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False,
                            num_workers=0)

    # to judge whether resume or not
    if args.resume:
        # Load checkpoint
        print("==> Resuming from checkpoint..")
        assert os.path.isdir(
            "checkpoint"), "Error:no checkpoint directory found!"
        # checkpoint = torch.load("./checkpoint/quick_draw_parameters")
        # checkpoint = torch.load("./checkpoint/quick_draw_parameters_resnet18")
        # net.load_state_dict(checkpoint['net'])
        # best_acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch']
        checkpoint = torch.load(
            "./checkpoint/quick_draw_parameters_after_train")
        net.load_state_dict(checkpoint['net'])
        start_epoch = 3
        print("reload parameters from {} is done".format(start_epoch))
        start_epoch += 1
        print("==> Resuming from epoch {} is working".format(start_epoch))

    # to choose the lost function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=5e-4)
    # to train data
    last_layer_names = ["module.classifier.0.weight",
                        "module.classifier.0.bias"]
    # last_layer_names = ['module.net.last_linear.weight', 'module.net.last_linear.bias',
    #                     'module.net.classifier.weight', 'module.net.classifier.bias']
    lr = 0.0000001

    for epoch in range(start_epoch, end_epoch):
        if epoch == 1:
            lr = 0.00012
        else:
            lr = lr / 10
        if patience == 2:
            patience = 0
            print("Loading checkpoint_best...........")
            checkpoint = torch.load("./checkpoint/quick_draw_parameters")
            net.load_state_dict(checkpoint['net'])
            lr = lr / 10.0
            print("the learning rate is changed to {}".format(lr))

        if epoch == 0:
            lr = 0.001
            for name, param in net.named_parameters():
                # print("the name is:{}".format(name))
                if (name not in last_layer_names):
                    param.requires_grad = False
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
        else:
            for param in net.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(
                net.parameters(), lr=lr, weight_decay=0.0001)
        print("the lr is {}".format(lr))
        # train for one epoch
        # if start_epoch == 2:
        #     test(val_loader, net, criterion, optimizer, epoch, args.print_freq)
        #     continue
        train(train_loader, net, criterion, optimizer, epoch, args.print_freq)
        print('Saving after train..')
        state = {
            'net': net.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/quick_draw_parameters_after_train')
        print('Saving after train is done')
        # evaluate on validation set
        test(val_loader, net, criterion, optimizer, epoch, args.print_freq)


# training
def train(train_loader, net, criterion, optimizer, epoch, print_freq):
    global log_path
    print("Epoch:{}".format(epoch))
    # switch to train mode
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # if batch_idx == 1:
        #     break
        # print("targets size is : {}".format(targets.size()))
        # print("batch_idx is {},(inputs:{}, targets:{}".format(batch_idx, inputs, targets))
        inputs, targets = inputs.to(device), targets.to(device)
        # compute the output
        outputs = net(inputs)

        # measure accuracy and record loss
        loss = criterion(outputs, targets)
        prec1 = accuracy(outputs.data, targets, topk=(1, 1))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        losses = train_loss / (batch_idx + 1)
        accuracy_temp = 100. * correct / total
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        if batch_idx % print_freq == 0:
            print('Epoch: [{}][{}/{}]'
                  'Loss: {} \t'
                  'accuracy: {}% \t'.format(epoch, batch_idx, len(train_loader), losses, accuracy_temp))
            with open(log_path, 'a') as logfile:
                print('Epoch: [{}][{}/{}]'
                      'Loss: {} \t'
                      'accuracy: {}% \t'.format(epoch, batch_idx, len(train_loader), losses, accuracy_temp), file=logfile)


# test
def test(val_loader, net, criterion, optimizer, epoch, print_freq):
    global best_acc
    global patience
    global min_loss
    print("Epoch:{}".format(epoch))
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # if batch_idx == 1:
            #     break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            losses = test_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            if batch_idx % print_freq == 0:
                print('Epoch: [{}][{}/{}]'
                      'Loss: {} \t'
                      'accuracy: {}% \t'.format(epoch, batch_idx, len(val_loader), losses, accuracy))
                with open(log_path, 'a') as logfile:
                    print('Epoch: [{}][{}/{}]'
                          'Loss: {} \t'
                          'accuracy: {}% \t'.format(epoch, batch_idx, len(val_loader), losses, accuracy), file=logfile)
            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # to change the patience
    if losses < min_loss:
        min_loss = losses
    else:
        patience += 1
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'min_loss': min_loss
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/quick_draw_parameters')
        best_acc = acc


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]


if __name__ == '__main__':
    main()
