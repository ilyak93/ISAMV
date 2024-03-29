import os
import time

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
        img_dir="/content/dance_data/train/",
        transform=True
    )
    val_set = CustomImageDataset(
        img_dir="/content/dance_data/test/",
        transform=False
    )

    from torch.utils.data import DataLoader

    batch_size = 4
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    input_channel = 3
    net = FADNet(input_channel=input_channel).cuda()

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

    gen_path = "/content/drive/MyDrive/Deep/"

    writer = SummaryWriter(gen_path)
    i = 0
    j = 0
    cycles = 2
    rows = 512
    cols = 640

    max_uint16 = pow(2, 16)

    previous_EPE = float(max_uint16)

    prev_cycles = 0

    prev_chkpnt = ""

    pretrain = False
    if pretrain:
        chkp = "epoch_199_loss_1.6628163054182723"
        PATH = gen_path + chkp
        checkpoint = torch.load(PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    duration = 1800  # 1 minute

    # Run the script for the specified duration
    start_time = time.time()

    for r in range(startRound, len(FADNet_loss_config["epoches"])):
        cycles = FADNet_loss_config["epoches"][r]
        for k in range(cycles):
            net.train()
            for [left_img, right_img], [target_disp, depths] in train_dataloader:
                left_img = left_img.unsqueeze(dim=1) / max_uint16
                right_img = right_img.unsqueeze(dim=1) / max_uint16

                actual_batch_size = left_img.size(0)
                row_indices_feature = torch.tensor(list(range(rows))).reshape([-1, 1]).repeat(1, cols).unsqueeze(
                    0).repeat(actual_batch_size, 1, 1).unsqueeze(1) / rows
                cols_indices_feature = torch.tensor(list(range(cols))).reshape([1, -1]).repeat(rows, 1).unsqueeze(
                    0).repeat(actual_batch_size, 1, 1).unsqueeze(1) / cols
                left_img_with_indices = torch.cat([left_img, row_indices_feature, cols_indices_feature], dim=1)
                right_img_with_indices = torch.cat([right_img, row_indices_feature, cols_indices_feature], dim=1)

                # target_disp = torch.sqrt(target_disp.unsqueeze(dim=1)).cuda()
                target_dis = depths.unsqueeze(dim=1).cuda() / 1000
                inputs = torch.cat((left_img_with_indices, right_img_with_indices), dim=1).cuda()
                output_net1, output_net2 = net(inputs)
                criterion = multiscaleloss(FADNet_loss_config['loss_scale'],
                                           1, FADNet_loss_config['loss_weights'][r],
                                           loss='L1', sparse=False)
                loss_net1 = criterion(output_net1, target_dis)
                loss_net2 = criterion(output_net2, target_dis)
                loss = loss_net1 + loss_net2
                output_net2_final = output_net2[0]
                flow2_EPE = EPE(output_net2_final, target_dis)

                train_losses.update(loss.data.item(), inputs.size(0))
                train_flow2_EPEs.update(flow2_EPE.data.item(), inputs.size(0))

                # compute gradient and do SGD step
                loss.backward()
                optimizer.step()

                if i % 10 == 0:
                    # print('Epoch: [{0}][{1}]\t'
                    #            'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    #            'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'.format(
                    #    r, i, loss=losses, flow2_EPE=flow2_EPEs))
                    writer.add_scalar("train/per_10_iterations/loss", loss.data.item(), i)
                    writer.add_scalar("train/per_10_iterations/EPE", flow2_EPE.data.item(), i)
                i = i + 1

            writer.add_scalar("train/epoch/loss", train_losses.avg, prev_cycles + k)
            writer.add_scalar("train/epoch/EPE", train_flow2_EPEs.avg, prev_cycles + k)

            # lr_scheduler.step()

            test_losses = AverageMeter()
            test_flow2_EPEs = AverageMeter()
            with torch.no_grad():
                test_losses = AverageMeter()
                test_flow2_EPEs = AverageMeter()
                net.eval()
                for [left_img, right_img], [target_disp, depths] in test_dataloader:
                    left_img = left_img.unsqueeze(dim=1) / max_uint16
                    right_img = right_img.unsqueeze(dim=1) / max_uint16

                    actual_batch_size = left_img.size(0)
                    row_indices_feature = torch.tensor(list(range(rows))).reshape([-1, 1]).repeat(1, cols).unsqueeze(
                        0).repeat(actual_batch_size, 1, 1).unsqueeze(1) / rows
                    cols_indices_feature = torch.tensor(list(range(cols))).reshape([1, -1]).repeat(rows, 1).unsqueeze(
                        0).repeat(actual_batch_size, 1, 1).unsqueeze(1) / cols
                    left_img_with_indices = torch.cat([left_img, row_indices_feature, cols_indices_feature], dim=1)
                    right_img_with_indices = torch.cat([right_img, row_indices_feature, cols_indices_feature], dim=1)

                    # target_disp = torch.sqrt(target_disp.unsqueeze(dim=1)).cuda()
                    target_dis = depths.unsqueeze(dim=1).cuda() / 1000
                    inputs = torch.cat((left_img_with_indices, right_img_with_indices), dim=1).cuda()
                    output_net1, output_net2 = net(inputs)

                    loss_net1 = EPE(output_net1, target_dis)
                    loss_net2 = EPE(output_net2, target_dis)
                    loss = loss_net1 + loss_net2
                    flow2_EPE = EPE(output_net2, target_dis)

                    test_losses.update(loss.data.item(), inputs.size(0))
                    test_flow2_EPEs.update(flow2_EPE.data.item(), inputs.size(0))

                    if j % 2 == 0:
                        # print('Epoch: [{0}][{1}]\t'
                        #            'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                        #            'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'.format(
                        #    r, i, loss=losses, flow2_EPE=flow2_EPEs))
                        writer.add_scalar("test/per_10_iterations/loss", loss.data.item(), j)
                        writer.add_scalar("test/per_10_iterations/EPE", flow2_EPE.data.item(), j)

                        indices = target_dis[0] != 0
                        diff = torch.zeros_like(target_dis[0]).cuda()
                        diff[indices] = torch.abs(target_dis[0][indices] - output_net2[0][indices]) / 256

                        diff_eq = torch.tensor(histogram_equalize(diff.cpu().numpy()))

                        output = torch.tensor(histogram_equalize((output_net2[0].cpu() / 256).numpy()))

                        label = torch.tensor(histogram_equalize((target_dis[0].cpu() / 256).numpy()))

                        orig_viz = torch.cat((left_img[0].cpu(),
                                              right_img[0].cpu(),
                                              output_net2[0].cpu() / 256,
                                              target_dis[0].cpu() / 256,
                                              output,
                                              label,
                                              diff.cpu(),
                                              diff_eq), 0).unsqueeze(1)

                        grid = torchvision.utils.make_grid(orig_viz)
                        writer.add_image(tag='Test_images/image_' + str(j % 13),
                                         img_tensor=grid, global_step=prev_cycles + k,
                                         dataformats='CHW')
                    j = j + 1

            writer.add_scalar("test/epoch/loss", test_losses.avg, prev_cycles + k)
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
            if time.time() - start_time >= duration:
                writer = SummaryWriter(gen_path)
                time.sleep(120)
                start_time = time.time()
            demo = open("/content/drive/MyDrive/Deep/demofile.txt", "w")
            demo.write("Ilya " + str(prev_cycles + k))
            demo.close()

        prev_cycles = prev_cycles + FADNet_loss_config["epoches"][max(0, r - 1)]