import torch
import torch.nn as nn
import torch.nn.functional as F

class IntrinsicsNet(nn.Module):
    def __init__(self):
        super(IntrinsicsNet, self).__init__()
        self.focal_x = torch.nn.Parameter(torch.tensor(833.85782381, requires_grad=True))
        self.focal_y = torch.nn.Parameter(torch.tensor(865.972622029, requires_grad=True))
        self.ccx = torch.nn.Parameter(torch.tensor(319.690452842080, requires_grad=True))
        self.ccy = torch.nn.Parameter(torch.tensor(200.489004453523, requires_grad=True))
        self.mov_x = torch.nn.Parameter((torch.tensor(-10., requires_grad=True)))
        self.mov_y = torch.nn.Parameter((torch.tensor(17.5, requires_grad=True)))
        self.R = torch.nn.Parameter(torch.eye(3, requires_grad=False))
        self.trans = torch.nn.Parameter(torch.tensor([-121.952452666493+250,
                                                  189.727110138437-300,
                                                  -813.867721109518],
                                  requires_grad=True).reshape([3,1]))
        self.width = 1280
        self.height = 720


    def forward(self, label):
        depthscale = 1000
        depth_image = label * depthscale
        U = torch.tensor(list(range(self.width)), requires_grad=False).reshape(1, -1).repeat_interleave(720, dim=0)
        V = torch.tensor(list(range(self.height)), requires_grad=False).reshape(-1, 1).repeat_interleave(1280, dim=1)
        Z = depth_image / depthscale
        X = (U - self.ccx) * Z / self.focal_x
        Y = (V - self.ccy) * Z / self.focal_y
        XYZ = torch.stack((X, Y, Z), axis=-1)
        indices = depth_image > 0
        xyz_matrix = torch.zeros_like(XYZ)
        xyz_matrix[:] = torch.nan
        xyz_matrix[indices, :] = XYZ[indices, :]
        real_indices = torch.argwhere(indices)
        point_u, point_v = real_indices.transpose(1,0)
        point_xyz = xyz_matrix[point_u, point_v].transpose(1,0)

        xyz = point_xyz * 1000
        # t[2] = t[2] + 150
        rotated_translated_xyz = self.R @ xyz + self.trans

        x, y = rotated_translated_xyz[0, :] / rotated_translated_xyz[2, :], \
               rotated_translated_xyz[1, :] / rotated_translated_xyz[2, :]

        pixel_u = x * self.focal_x + self.ccx + self.mov_x  # + is right
        pixel_v = y * self.focal_y + self.ccy + self.mov_y  # + is down
        pixel_u[(pixel_u <= 0) | (pixel_u > 1279.5)] = 0
        pixel_v[(pixel_v <= 0) | (pixel_v > 719.5)] = 0

        pixel_u, pixel_v = pixel_v, pixel_u
        #here is the error
        igul_u_left, igul_v_left = torch.round(pixel_u).long(), torch.round(pixel_v).int().long()
        new_label = torch.zeros((720, 1280), dtype=torch.float32)
        new_label[igul_u_left, igul_v_left] = depth_image[point_u, point_v]

        return new_label