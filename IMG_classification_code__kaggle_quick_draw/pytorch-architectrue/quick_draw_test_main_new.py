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
from quick_draw_test_dataset import quick_draw_test_dataset
import csv

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description="quick_draw_test")
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


preprocess_tencrop = transforms.Compose([
    transforms.Resize(image_size),
    transforms.TenCrop(input_size),
    transforms.Lambda(lambda crops: torch.stack(
        [temp_trans(crop)for crop in crops])),
    # transforms.ToTensor(),
    # normalize
])


def main():
    TTA10_preprocess = [preprocess_tencrop]
    args = parser.parse_args()
    print("=> creating the model '{}'".format(args.arch))
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
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    print("==> creat model is done !")

    print("=> reload parameters from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    # checkpoint = torch.load("./checkpoint/quick_draw_parameters")
    checkpoint = torch.load("./checkpoint/checkpoint1127_best.pth.tar")
    net.load_state_dict(checkpoint['state_dict'])
    print("reloading the parameters is done!")

    print("==> Preparing the test data...")
    test_root = "/root/hdd/yankun/Kaggle/quick_draw/all"
    test_source = "test_meta_file"
    test_dataset = quick_draw_test_dataset(
        test_root,
        test_source,
        transforms.Compose([
            transforms.ToTensor()
        ]))
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=0)
    print("===> preparing the data is done")

    # image_map_path = "/root/hdd/yankun/Kaggle/quick_draw/all/image_num"
    image_map_path = "/root/hdd/yankun/Kaggle/quick_draw/all/image_num_new_new_all"

    image_map = []
    with open(image_map_path) as image_map_file:
        image_map_lines = image_map_file.readlines()
    for line in image_map_lines:
        image_str, image_num = line.rstrip().split()
        image_num = int(image_num)
        image_map.append(image_str)
    # leng = len(image_map)
    # for i in range(leng):
    #     print(image_map[i], i)
    names = ['key_id', 'word']
    submission_path = "/root/hdd/yankun/Kaggle/quick_draw/all/submission.csv"
    with open(submission_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(names)
        num = 0
        for batch_idx, (inputs, key_id) in enumerate(test_loader):
            len_loader = len(test_loader)
            result_list = []
            # num += 1
            print("==>the batch_idx : {} is working..".format(batch_idx))
            inputs = inputs.to(device)
            outputs = net(inputs)
            # print("key_id size is : {}".format(key_id.shape))
            # print("outputs size is : {}".format(outputs.shape))
            _, image_id = outputs.sort(1, descending=True)
            # print("image_id size is {}, and type is {}".format(
            #     image_id.shape, type(image_id)))
            # print(image_id)
            image_id_numpy = image_id.cpu().detach().numpy()
            key_id_numpy = key_id.detach().numpy()
            # print(key_id.type())
            # print(image_id_numpy)
            # print(image_id_numpy.shape[0])
            for i in range(image_id_numpy.shape[0]):
                str1 = image_map[image_id_numpy[i][0]]
                str2 = image_map[image_id_numpy[i][1]]
                str3 = image_map[image_id_numpy[i][2]]
                # str = "'" + str1 + "'" + ' ' + "'" + str2 + "'" + ' ' + "'" + str3 + "'"
                str = str1 + ' ' + str2 + ' ' + str3
                im_num = key_id_numpy[i]
                compent = (im_num, str)
                result_list.append(compent)
            writer.writerows(result_list)
            print("==>the batch_idx : {}/{} is done..".format(batch_idx, len_loader))
            # if num > 2:
            #     break


if __name__ == '__main__':
    main()
