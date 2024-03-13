import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lenet import LeNet
from alexnet import AlexNet
from vggnet import vgg
from googlenet import GoogLeNet
from resnet import resnet34
from densenet import densenet121
from mobilenet_v2 import MobileNetV2
from mobilenet_v3 import MobileNetV3
from shufflenet import shufflenet_v2_x1_0
from efficientnet import efficientnet_b0
from efficientnetv2 import efficientnetv2_s
from convnext import convnext_tiny


def main(args):

    ################################################################################
    #
    #                           tensorboard
    #
    ################################################################################

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(log_dir="runs")

    ################################################################################
    #
    #                           GPU
    #
    ################################################################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    ################################################################################
    #
    #                           load data
    #
    ################################################################################

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.CIFAR10(root='/data0/CIFAR', train=True, download=False, transform=transform)
    validate_dataset = torchvision.datasets.CIFAR10(root='/data0/CIFAR', train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=nw)
    print("using {} images for training, {} images for validation.".format(len(train_dataset),len(validate_dataset)))

    ################################################################################
    #
    #                           load model
    #
    ################################################################################

    if args.model_name == "lenet":
        net = LeNet()

    elif args.model_name == "alexnet":
        net = AlexNet(num_classes=args.num_classes, init_weights=True)
        
    elif args.model_name == "vgg16":
        net = vgg(model_name=args.model_name, num_classes=args.num_classes, init_weights=True)
        
    elif args.model_name == "googlenet":
        net = GoogLeNet(num_classes=args.num_classes, aux_logits=True, init_weights=True)
        # 如果要使用官方的预训练权重，注意是将权重载入官方的模型，不是自己实现的模型
        # 官方的模型中使用了bn层以及改了一些参数，不能混用
        # import torchvision
        # net = torchvision.models.googlenet(num_classes=5)
        # model_dict = net.state_dict()
        # # 预训练权重下载地址: https://download.pytorch.org/models/googlenet-1378be20.pth
        # pretrain_model = torch.load("googlenet.pth")
        # del_list = ["aux1.fc2.weight", "aux1.fc2.bias",
        #             "aux2.fc2.weight", "aux2.fc2.bias",
        #             "fc.weight", "fc.bias"]
        # pretrain_dict = {k: v for k, v in pretrain_model.items() if k not in del_list}
        # model_dict.update(pretrain_dict)
        # net.load_state_dict(model_dict)


    elif args.model_name == "resnet34":
        net = resnet34()
        # load pretrain weights, download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
        model_weight_path = "./resnet34-pre.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
        # for param in net.parameters():
        #     param.requires_grad = False
        # change fc layer structure
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, args.num_classes)


    elif args.model_name == "densenet":
        model = densenet121(num_classes=args.num_classes).to(device)
        # densenet121 官方权重下载地址: https://download.pytorch.org/models/densenet121-a639ec97.pth
        model_weight_path = "./densenet121.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
        if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)


    elif args.model_name == "mobilenetv2":
        net = MobileNetV2(num_classes=args.num_classes)
        # load pretrain weights, download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
        model_weight_path = "./mobilenet_v2.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))


    elif args.model_name == "shufflenetv2":
        model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)
        # densenet121 官方权重下载地址: https://download.pytorch.org/models/densenet121-a639ec97.pth
        model_weight_path = "./shufflenetv2_x1.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
        if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)


    elif args.model_name == "efficientnet":
        model = efficientnet_b0(num_classes=args.num_classes).to(device)
        model_weight_path = "./efficientnetb0.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
        if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)


    elif args.model_name == "convnext":
        model = convnext_tiny(num_classes=args.num_classes).to(device)
        model_weight_path = "./convnext_tiny_1k_224_ema.pth"
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
        

    ################################################################################
    #
    #                           optimizer 
    #
    ################################################################################

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    pg = [para for para in model.parameters() if para.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    ################################################################################
    #
    #                           train and test
    #
    ################################################################################

    best_acc = 0.0
    save_path = './{}.pth'.format(args.model_name)
    train_steps = len(train_loader)
    for epoch in range(args.epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        # for step, data in enumerate(train_loader, start=0):
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            if args.model_name == "googlenet":
                logits, aux_logits2, aux_logits1 = net(images.to(device))
                loss0 = loss_function(logits, labels.to(device))
                loss1 = loss_function(aux_logits1, labels.to(device))
                loss2 = loss_function(aux_logits2, labels.to(device))
                loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            else:
                outputs = net(images.to(device))
                loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, args.epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], running_loss / train_steps, epoch)
        tb_writer.add_scalar(tags[1], val_accurate, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')



if __name__ == '__main__':

    ################################################################################
    #
    #                           parameter
    #
    ################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--model_name', type=str, default="lenet")   
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()

    main(args)
