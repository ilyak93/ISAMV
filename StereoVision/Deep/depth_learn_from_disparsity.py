import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.autograd import Variable

from networks.Autoencoder import ConvAutoencoder
from networks.Autoencoder_CIFAR import Autoencoder
from networks.UNet import UNet

from dataloaders import CustomImageDataset


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



if __name__ == '__main__':
    torch.manual_seed(13)


    dataset = CustomImageDataset(
            img_dir="C:/dataset/small_correct/"
        )

    test_size = 50
    lengths = [len(dataset)-test_size, test_size]
    train_set, val_set = torch.utils.data.random_split(dataset, lengths)

    from torch.utils.data import DataLoader

    batch_size = 4
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)


    beta = 0.999
    momentum = 0.9
    lr = 0.0001

    class Net(nn.Module):
        def __init__(self, initial_guess):
            super(Net, self).__init__()
            focal_baseline = nn.Parameter(initial_guess.type(torch.float64), requires_grad=True)
            self.register_parameter(name='focal_baseline', param=focal_baseline)

        def forward(self, input):
            y = self.focal_baseline / input
            return y, self.focal_baseline.data.item()


    [left_samples, right_samples], [dis, depths] = next(iter(train_dataloader))
    #net = Net((dis[0] * depths[0]).mean() / 1000).cuda()
    net = ConvAutoencoder().cuda()


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr,
                                            betas=(momentum, beta), amsgrad=True)

    epochs = 1000
    writer = SummaryWriter()
    k = 0
    j = 0
    max_uint16 = pow(2, 16)
    for i in range(epochs):
        net.train()
        train_losses = AverageMeter()
        for [left_samples, right_samples], [dis, depths] in train_dataloader:
            depths = depths.cuda().unsqueeze(1) / 1000
            dis = dis.cuda().unsqueeze(1)
            target_valid = (depths != 0) & (dis != 0)
            dis_indices = dis != 0
            dis_inv_values = 1 / dis[dis_indices]
            dis_inv = torch.zeros_like(dis)
            dis_inv[dis_indices] = dis_inv_values
            #output, param = net(dis[target_valid])
            output = net(dis_inv)
            #loss = F.smooth_l1_loss(output[target_valid], depths[target_valid], size_average=True)
            loss = F.smooth_l1_loss(output[target_valid], depths[target_valid], size_average=True)
            loss.backward()
            optimizer.step()

            if j % 10 == 0:
                # print('Epoch: [{0}][{1}]\t'
                #            'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                #            'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'.format(
                #    r, i, loss=losses, flow2_EPE=flow2_EPEs))
                writer.add_scalar("train/per_10_iterations/loss", loss.data.item(), j)
                train_losses.update(loss.data.item(), batch_size)
                #writer.add_scalar("train/per_10_iterations/param", param, j)
            j = j + 1
        writer.add_scalar("train/epoch/loss", train_losses.avg, i)
        #writer.add_scalar("train/epoch/param", param, i)

        net.eval()

        test_losses = AverageMeter()
        with torch.no_grad():
            for [left_samples, right_samples], [dis, depths] in test_dataloader:
                depths = depths.cuda().unsqueeze(1) / 1000
                dis = dis.cuda().unsqueeze(1)
                target_valid = (depths != 0) & (dis != 0)
                dis_indices = dis != 0
                dis_inv_values = 1 / dis[dis_indices]
                dis_inv = torch.zeros_like(dis)
                dis_inv[dis_indices] = dis_inv_values
                # output, param = net(dis[target_valid])
                output = net(dis_inv)
                # loss = F.smooth_l1_loss(output[target_valid], depths[target_valid], size_average=True)
                loss = F.smooth_l1_loss(output[target_valid], depths[target_valid], size_average=True)
                new_output = torch.zeros_like(dis)
                new_output = output

                if k % 2 == 0:
                    # print('Epoch: [{0}][{1}]\t'
                    #            'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    #            'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'.format(
                    #    r, i, loss=losses, flow2_EPE=flow2_EPEs))
                    writer.add_scalar("test/per_10_iterations/loss", loss.data.item(), k)
                    orig_viz = torch.cat((dis[0].cpu(), new_output[0].cpu(), depths[0].cpu()), 0).unsqueeze(1)
                    grid = torchvision.utils.make_grid(orig_viz)
                    writer.add_image(tag='Test_images/image_' + str(k % 13),
                                     img_tensor=grid, global_step=i,
                                     dataformats='CHW')
                    test_losses.update(loss.data.item(), batch_size)
                k = k + 1
            writer.add_scalar("test/epoch/loss", test_losses.avg, i)
