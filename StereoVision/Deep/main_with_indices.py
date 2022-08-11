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

from torch.utils.tensorboard import SummaryWriter

#tensorboard --logdir runs

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

    dataset = CustomImageDataset(
        img_dir="E:/dataset/left_right_dis_dep/"
    )
    lengths = [3300, len(dataset)-3300]
    train_set, val_set = torch.utils.data.random_split(dataset, lengths)

    from torch.utils.data import DataLoader

    batch_size = 3
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
    test_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=3)

    net = FADNet(input_channel=3).cuda()

    FADNet_loss_config = {
        "loss_scale":7,
        "round":4,
        "loss_weights":[[0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005],
                        [0.6, 0.32, 0.08, 0.04, 0.02, 0.01, 0.005],
                        [0.8, 0.16, 0.04, 0.02, 0.01, 0.005, 0.0025],
                        [1.0, 0., 0., 0., 0., 0., 0.]],
        "epoches":[20, 20, 20, 30]
    }
    startRound, train_round = 0, 8

    beta = 0.999
    momentum = 0.9
    lr = 0.00001
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr,
                                            betas=(momentum, beta), amsgrad=True)
    decayRate = 0.96
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    train_losses = AverageMeter()
    train_flow2_EPEs = AverageMeter()

    writer = SummaryWriter()
    i = 0
    j = 0
    cycles = 2
    rows = 512
    cols = 640

    for r in range(startRound, train_round, 2):
        for k in range(cycles):
            for [left_img, right_img], [target_disp, depths] in train_dataloader:
                left_img = left_img.unsqueeze(dim=1)
                right_img = right_img.unsqueeze(dim=1)
                actual_batch_size = left_img.size(0)
                row_indices_feature = torch.tensor(list(range(rows))).reshape([-1, 1]).repeat(1, cols).unsqueeze(
                    0).repeat(actual_batch_size, 1, 1).unsqueeze(1)
                cols_indices_feature = torch.tensor(list(range(cols))).reshape([1, -1]).repeat(rows, 1).unsqueeze(
                    0).repeat(actual_batch_size, 1, 1).unsqueeze(1)
                left_img_with_indices = torch.cat([left_img, row_indices_feature, cols_indices_feature], dim=1)
                right_img_with_indices = torch.cat([right_img, row_indices_feature, cols_indices_feature], dim=1)

                target_disp = torch.sqrt(target_disp.unsqueeze(dim=1)).cuda()
                inputs = torch.cat((left_img_with_indices, right_img_with_indices), dim=1).cuda()
                output_net1, output_net2 = net(inputs)
                criterion = multiscaleloss(FADNet_loss_config['loss_scale'], 1, FADNet_loss_config['loss_weights'][int(r/cycles)], loss='L1', sparse=False)
                loss_net1 = criterion(output_net1, target_disp)
                loss_net2 = criterion(output_net2, target_disp)
                loss = loss_net1 + loss_net2
                output_net2_final = output_net2[0]
                flow2_EPE = EPE(output_net2_final, target_disp)

                train_losses.update(loss.data.item(), inputs.size(0))
                train_flow2_EPEs.update(flow2_EPE.data.item(), inputs.size(0))

                # compute gradient and do SGD step
                loss.backward()
                optimizer.step()

                if i % 10 == 0:
                    #print('Epoch: [{0}][{1}]\t'
                    #            'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    #            'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'.format(
                    #    r, i, loss=losses, flow2_EPE=flow2_EPEs))
                    writer.add_scalar("train/per_10_iterations/loss", loss.data.item(), i)
                    writer.add_scalar("train/per_10_iterations/EPE", flow2_EPE.data.item(), i)
                i = i + 1

            writer.add_scalar("train/epoch/loss", train_losses.avg, r+k)
            writer.add_scalar("train/epoch/EPE", train_flow2_EPEs.avg, r+k)

            lr_scheduler.step()

            test_losses = AverageMeter()
            test_flow2_EPEs = AverageMeter()

            for [left_img, right_img], [target_disp, depths] in test_dataloader:
                left_img = left_img.unsqueeze(dim=1)
                right_img = right_img.unsqueeze(dim=1)

                actual_batch_size = left_img.size(0)
                row_indices_feature = torch.tensor(list(range(rows))).reshape([-1, 1]).repeat(1, cols).unsqueeze(
                    0).repeat(actual_batch_size, 1, 1).unsqueeze(1)
                cols_indices_feature = torch.tensor(list(range(cols))).reshape([1, -1]).repeat(rows, 1).unsqueeze(
                    0).repeat(actual_batch_size, 1, 1).unsqueeze(1)
                left_img_with_indices = torch.cat([left_img, row_indices_feature, cols_indices_feature], dim=1)
                right_img_with_indices = torch.cat([right_img, row_indices_feature, cols_indices_feature], dim=1)

                target_disp = torch.sqrt(target_disp.unsqueeze(dim=1)).cuda()
                inputs = torch.cat((left_img_with_indices, right_img_with_indices), dim=1).cuda()
                output_net1, output_net2 = net(inputs)
                criterion = multiscaleloss(FADNet_loss_config['loss_scale'], 1, FADNet_loss_config['loss_weights'][int(r/cycles)], loss='L1', sparse=False)
                loss_net1 = criterion(output_net1, target_disp)
                loss_net2 = criterion(output_net2, target_disp)
                loss = loss_net1 + loss_net2
                output_net2_final = output_net2[0]
                flow2_EPE = EPE(output_net2_final, target_disp)

                test_losses.update(loss.data.item(), inputs.size(0))
                test_flow2_EPEs.update(flow2_EPE.data.item(), inputs.size(0))

                if j % 10 == 0:
                    #print('Epoch: [{0}][{1}]\t'
                    #            'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    #            'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'.format(
                    #    r, i, loss=losses, flow2_EPE=flow2_EPEs))
                    writer.add_scalar("test/per_10_iterations/loss", loss.data.item(), j)
                    writer.add_scalar("test/per_10_iterations/EPE", flow2_EPE.data.item(), j)
                    orig_viz = torch.cat(((left_img[0] / left_img[0].max()), (right_img[0] / right_img[0].max()), target_disp[0].cpu(), torch.square(target_disp[0]).cpu(), output_net2_final[0].cpu()), 0).unsqueeze(1)
                    grid = torchvision.utils.make_grid(orig_viz)
                    writer.add_image(tag='Test_images/image_' + str(j),
                                     img_tensor=grid, global_step=r+k,
                                     dataformats='CHW')
                j = j + 1

            writer.add_scalar("test/epoch/loss", test_losses.avg, r+k)
            writer.add_scalar("test/epoch/EPE", test_flow2_EPEs.avg, r+k)