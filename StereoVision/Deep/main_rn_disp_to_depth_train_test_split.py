import os

import matplotlib
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from dataloaders import CustomImageDataset
from losses.multiscaleloss import multiscaleloss, EPE
from networks.FADNet import FADNet
import torchvision.transforms as T
from torch import nn
from torchvision.models.segmentation import fcn_resnet101

from torch.utils.tensorboard import SummaryWriter

from skimage import exposure
import numpy as np

def histogram_equalize(img):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf).astype(np.float32)

# tensorboard --logdir runs

if __name__ == '__main__':
    torch.manual_seed(13)

    print(torch.cuda.is_available())


    class AverageMeter(object):

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


    train_set = CustomImageDataset(
        img_dir="/content/data/train/",
        transform=True
    )
    val_set = CustomImageDataset(
        img_dir="/content/data/test/",
        transform=False
    )

    from torch.utils.data import DataLoader

    batch_size = 4
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    in_channels = 1
    net = fcn_resnet101(pretrained=True)
    backbone_first_conv = net.backbone.conv1
    net.backbone.conv1 = nn.Conv2d(in_channels=in_channels,
                                   out_channels=backbone_first_conv.out_channels,
                                   kernel_size=backbone_first_conv.kernel_size,
                                   stride=backbone_first_conv.stride,
                                   padding=backbone_first_conv.padding,
                                   bias=backbone_first_conv.bias)

    num_classes = 1  # Adjust this according to the number of output channels you want
    classifier_last_layer = net.classifier[4]
    net.classifier[4] = nn.Conv2d(in_channels=classifier_last_layer.in_channels,
                                  out_channels=num_classes,
                                  kernel_size=classifier_last_layer.kernel_size,
                                  stride=classifier_last_layer.stride)
    net = net.cuda()
    FADNet_loss_config = {
        "loss_scale": 7,
        "round": 4,
        "loss_weights": [[0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005],
                         [0.6, 0.32, 0.08, 0.04, 0.02, 0.01, 0.005],
                         [0.8, 0.16, 0.04, 0.02, 0.01, 0.005, 0.0025],
                         [1.0, 0., 0., 0., 0., 0., 0.]],
        "epoches": [30, 30, 30, 4000]
    }
    startRound, train_round = 0, 8

    beta = 0.999
    momentum = 0.9
    lr = 0.0000005
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr,
                                 betas=(momentum, beta), amsgrad=True)
    decayRate = 0.98
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    train_losses = AverageMeter()
    train_flow2_EPEs = AverageMeter()

    writer = SummaryWriter("/content/drive/MyDrive/Deep/")
    
    i = 0
    j = 0
    cycles = 2
    rows = 512
    cols = 640

    max_uint16 = pow(2, 16)

    previous_EPE = float(max_uint16)

    prev_cycles = 0

    prev_chkpnt = ""

    for r in range(startRound, len(FADNet_loss_config["epoches"])):
        cycles = FADNet_loss_config["epoches"][r]
        for k in range(cycles):
            net.train()
            for [left_img, right_img], [target_disp, depths] in train_dataloader:
                
                target_disp = torch.sqrt(target_disp).unsqueeze(dim=1)
                target_disp /= 2 ** 8


                # target_disp = torch.sqrt(target_disp.unsqueeze(dim=1)).cuda()
                label = depths.unsqueeze(dim=1).cuda() / 1000
                inputs = target_disp.cuda()
                output = net(inputs)["out"]

                flow2_EPE = EPE(output, label)

                train_flow2_EPEs.update(flow2_EPE.data.item(), inputs.size(0))

                # compute gradient and do SGD step
                flow2_EPE.backward()
                optimizer.step()

                if i % 10 == 0:
                    # print('Epoch: [{0}][{1}]\t'
                    #            'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    #            'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'.format(
                    #    r, i, loss=losses, flow2_EPE=flow2_EPEs))
                    writer.add_scalar("train/per_10_iterations/EPE", flow2_EPE.data.item(), i)
                i = i + 1

            writer.add_scalar("train/epoch/EPE", train_flow2_EPEs.avg, prev_cycles + k)

            # lr_scheduler.step()

            test_flow2_EPEs = AverageMeter()
            with torch.no_grad():
                test_flow2_EPEs = AverageMeter()
                net.eval()
                for [left_img, right_img], [target_disp, depths] in test_dataloader:
                    target_disp = torch.sqrt(target_disp).unsqueeze(dim=1)
                    target_disp /= 2 ** 8

                    # target_disp = torch.sqrt(target_disp.unsqueeze(dim=1)).cuda()
                    label = depths.unsqueeze(dim=1).cuda() / 1000
                    inputs = target_disp.cuda()
                    output = net(inputs)["out"]

                    flow2_EPE = EPE(output, label)

                    test_flow2_EPEs.update(flow2_EPE.data.item(), inputs.size(0))

                    if j % 2 == 0:
                        # print('Epoch: [{0}][{1}]\t'
                        #            'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                        #            'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'.format(
                        #    r, i, loss=losses, flow2_EPE=flow2_EPEs))
                        writer.add_scalar("test/per_10_iterations/EPE", flow2_EPE.data.item(), j)



                        
                        eq_disp = torch.tensor(histogram_equalize(inputs[0].cpu().numpy()))
                        
                        orig_viz = torch.cat((left_img[0].cpu().unsqueeze(0) / 2 ** 16,
                                              right_img[0].cpu().unsqueeze(0) / 2 ** 16,
                                              eq_disp,
                                              output[0].cpu() / 256,
                                              label[0].cpu() / 256,
                                              torch.abs(label[0].cpu() / 256 -
                                                        output[0].cpu() / 256)),
                                             0).unsqueeze(1)
                        grid = torchvision.utils.make_grid(orig_viz)
                        writer.add_image(tag='Test_images/image_' + str(j % 13),
                                         img_tensor=grid, global_step=prev_cycles + k,
                                         dataformats='CHW')
                    j = j + 1

            writer.add_scalar("test/epoch/EPE", test_flow2_EPEs.avg, prev_cycles + k)
            if test_flow2_EPEs.avg < previous_EPE:
                if prev_chkpnt != '':
                    open(prev_chkpnt, "w").close()
                    os.remove(prev_chkpnt)
                torch.save({
                    'epoch': prev_cycles + k,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_flow2_EPEs.avg,
                }, "/content/drive/MyDrive/Deep/" + "epoch_" + str(prev_cycles + k) + "_loss_" + str(
                    test_flow2_EPEs.avg))
                previous_EPE = test_flow2_EPEs.avg
                prev_chkpnt = "/content/drive/MyDrive/Deep/" + "epoch_" + str(prev_cycles + k) + "_loss_" + str(
                    test_flow2_EPEs.avg)
            demo = open("/content/drive/MyDrive/Deep/demofile.txt", "w")
            demo.write("Ilya " + str(prev_cycles + k))
            demo.close()

        prev_cycles = prev_cycles + FADNet_loss_config["epoches"][max(0, r - 1)]
