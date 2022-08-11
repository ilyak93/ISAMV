
from dataloaders import CustomImageDataset
from losses.multiscaleloss import multiscaleloss, EPE
from networks.FADNet import FADNet

from torch.utils.tensorboard import SummaryWriter
import matplotlib
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

#tensorboard --logdir runs
from utils.preprocess import scale_disp


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1,3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
    '''

    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')



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
        img_dir="C:/dataset/small_lrdd/"
    )
    test_size = 50
    lengths = [len(dataset)-test_size, test_size]
    train_set, val_set = torch.utils.data.random_split(dataset, lengths)

    from torch.utils.data import DataLoader

    batch_size = 4
    test_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    batch_size = 4
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4)

    input_channel = 1
    net = FADNet(input_channel=input_channel).cuda()

    checkpoint = torch.load("finished_runs/disp_no_mr_best/epoch_501_loss_2.784626865386963",
                            map_location=torch.device('cuda'))
    net.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    depth_eval = False
    disp = False

    i = 0
    j = 0
    rows = 512
    cols = 640
    max_uint16 = pow(2,16)
    index = 36
    with torch.no_grad():
        test_losses = AverageMeter()
        test_flow2_EPEs = AverageMeter()
        net.eval()
        itr = iter(train_dataloader)
        [left_img, right_img], [target_disp, depths] = next(itr)
        [left_img, right_img], [target_disp, depths] = next(itr)

        #for depth eval: depth / 1000
        #for disparity eval: sqrt(disparity)
        #for fixed disparity eval: no transform

        left_img = left_img.unsqueeze(dim=1) / max_uint16
        right_img = right_img.unsqueeze(dim=1) / max_uint16
        #target_dis = depths.unsqueeze(dim=1).cuda() / 1000
        target_dis = torch.sqrt(target_disp.unsqueeze(dim=1).cuda())

        inputs = torch.cat((left_img, right_img), dim=1).cuda()
        output_net1, output_net2 = net(inputs)

        output_net1_depth = torch.zeros_like(output_net1)
        output_net2_depth = torch.zeros_like(output_net2)
        gt = torch.zeros_like(target_dis)

        if disp:
            output_net1_depth[output_net1 != 0.] = 470. / output_net1[output_net1 != 0.]
            output_net2_depth[output_net2 != 0.] = 470. / output_net2[output_net2 != 0.]
            gt[target_dis != 0.] = 470. / target_dis[target_dis != 0.]
            output_net1 = output_net1_depth
            output_net2 = output_net2_depth
            target_dis = gt

        loss_net1 = EPE(output_net1, target_dis)
        loss_net2 = EPE(output_net2, target_dis)
        loss = loss_net1 + loss_net2
        flow2_EPE = EPE(output_net2, target_dis)

        print("output hist")
        out = output_net2[0].cpu().detach().squeeze().numpy()
        output_net2[0] = output_net2[0]
        out_hist = torch.histc(output_net2[0][target_dis[0] != 0].squeeze().cpu())
        bins = 100
        x = range(bins)
        plt.bar(x, out_hist, align='center')
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.show()
        plt.imshow(out)
        plt.show()

        print("label hist")
        disp = target_dis[0].cpu().detach().squeeze().numpy()
        #target_dis[0] = target_dis[0] + torch.normal(mean=0, std=5, size=(1,512,640)).cuda()
        out_hist = torch.histc((target_dis[0].squeeze()).cpu()[target_dis[0].squeeze() != 0])
        bins = 100
        x = range(bins)
        plt.bar(x, out_hist, align='center')
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.show()
        plt.imshow(disp)
        plt.show()

        print("diff hist")
        indices = target_dis[0] != 0
        disp = 470 / target_dis[0][indices]
        dif = torch.abs((target_dis[0][indices] - output_net2[0][indices].cuda()))
        out_hist = torch.histc(dif.squeeze().cpu())
        bins = 100
        x = range(bins)
        plt.bar(x, out_hist, align='center')
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.show()

        if depth_eval:
            thermal_frame = left_img[0].cpu().detach().squeeze().numpy()
            thermal_frame = ( thermal_frame - thermal_frame.min() ) / thermal_frame.max()
            thermal_frame = np.round(thermal_frame * 255).astype(np.uint8)
            thermal_frame = np.stack((thermal_frame,thermal_frame,thermal_frame), axis=2)
            #plt.imshow(thermal_frame)
            #plt.show()
            depth_frame = target_dis[0].cpu().detach().squeeze().numpy()
            frame_name = '0'
            mask = depth_frame != 0

            color_raw = o3d.geometry.Image(thermal_frame)
            depth_raw = o3d.geometry.Image(depth_frame)

            '''
            depth_raw = o3d.geometry.Image(np.array(cv2.imread("0depth_aligned.png", cv2.IMREAD_ANYDEPTH)))
            plt.imshow(depth_raw)
            plt.show()
            dd = np.asarray(depth_raw)
            '''

            therm_depth_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_raw, depth_raw, depth_scale=1000.0, depth_trunc=1000, convert_rgb_to_intensity=False)
            '''
            print(rgbd_image)
    
            plt.subplot(1, 2, 1)
            plt.title('Redwood grayscale image')
            plt.imshow(rgbd_image.color)
            plt.subplot(1, 2, 2)
            plt.title('Redwood depth image')
            plt.imshow(rgbd_image.depth)
            plt.show()
            '''
            # intrinsic = read_pinhole_camera_intrinsic("real_sense_intrinsic")
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                          fx=1013.03991699219,
                                                          fy=1013.03991699219,
                                                          cx=655.809936523438,
                                                          cy=380.619934082031)

            '''
            depth intrinsics:
                Width:        1280
                Height:       720
                PPX:          655.809936523438
                PPY:          380.619934082031
                Fx:           1013.03991699219
                Fy:           1013.03991699219
    
            color intrinsics:
                Width:        1280
                Height:       720
                PPX:          648.483276367188
                PPY:          359.049194335938
                Fx:           787.998596191406
                Fy:           786.333374023438
            '''
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                therm_depth_image, intrinsic)
            # Flip it, otherwise the pointcloud will be upside down
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # pcd.cluster_dbscan(eps=20, min_points=40, print_progress=True)
            # o3d.visualization.draw_geometries([pcd])

            colors_org = np.asarray(pcd.colors)
            colors_org = (colors_org - colors_org.min()) / colors_org.max()
            colors_org = np.round(colors_org * 255).astype(np.uint8)
            points_org = np.asarray(pcd.points)
            # xmin = points_org.min(axis=0)
            # xmax = points_org.max(axis=0)
            # print()

            output_file = './' + frame_name + ".ply"
            create_output(points_org, colors_org, output_file)

            thermal_frame = left_img[0].cpu().detach().squeeze().numpy()
            plt.imshow(thermal_frame)
            plt.show()

            pcd_rs = o3d.io.read_point_cloud(output_file)
            o3d.visualization.draw_geometries([pcd_rs])


            depth_frame = output_net2[0].cpu().detach().squeeze().numpy()
            depth_frame = depth_frame * mask
            print("depth show")
            plt.imshow(depth_frame)
            plt.show()

            frame_name = '1'

            color_raw = o3d.geometry.Image(thermal_frame)
            depth_raw = o3d.geometry.Image(depth_frame)



            therm_depth_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_raw, depth_raw, depth_scale=1000.0, depth_trunc=1000, convert_rgb_to_intensity=False)


            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                therm_depth_image, intrinsic)
            # Flip it, otherwise the pointcloud will be upside down
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # pcd.cluster_dbscan(eps=20, min_points=40, print_progress=True)
            # o3d.visualization.draw_geometries([pcd])

            colors_org = np.asarray(pcd.colors)
            colors_org = (colors_org - colors_org.min()) / colors_org.max()
            colors_org = np.round(colors_org * 255).astype(np.uint8)
            points_org = np.asarray(pcd.points)
            # xmin = points_org.min(axis=0)
            # xmax = points_org.max(axis=0)
            # print()

            output_file = './' + frame_name + ".ply"
            create_output(points_org, colors_org, output_file)

            thermal_frame = left_img[0].cpu().detach().squeeze().numpy()
            plt.imshow(thermal_frame)
            plt.show()

            pcd_rs = o3d.io.read_point_cloud(output_file)
            #points = np.asarray(pcd_rs.points)
            #pcd_rs = pcd.select_by_index(np.where(points[:, 1] > 60)[0])
            o3d.visualization.draw_geometries([pcd_rs])

            #points = np.asarray(pcd_rs.points)
            #mn = points.min(axis=0)
            #mx = points.max(axis=0)
