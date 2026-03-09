"""基于 nvdiffrast 的网格渲染器。

支持：
- 顶点属性插值渲染
- 由深度/视差构建规则网格
- 基于深度边缘的可见性掩码
"""


import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr


class RGBDRenderer:
    def __init__(self, device):
        self.device = device
        self.eps = 1e-4
        self.near_z = 1e-4
        self.far_z = 1e4
        self.glctx = None
    
    def render_mesh(self, mesh_dict, cam_int, cam_ext):
        '''
        input:
            mesh: the output for construct_mesh function
            cam_int: [b,3,3] normalized camera intrinsic matrix
            cam_ext: [b,3,4] camera extrinsic matrix with the same scale as depth map

            camera coord: x to right, z to front, y to down
        
        output:
            render: [b,3,h,w]
        '''
        # mesh_dict 中 attributes 可携带任意插值属性（此工程用 xyz+mask）。
        vertice = mesh_dict["vertice"]  # [b,h*w,3]
        faces = mesh_dict["faces"]  # [nface,3]
        attributes = mesh_dict["attributes"]  # [b,h*w,4]
        h, w = mesh_dict["size"]

        ############
        # to NDC space
        # 世界坐标 -> 相机坐标 -> 齐次裁剪空间。
        vertice_homo = self.lift_to_homo(vertice)  # [b,h*w,4]
        # [b,1,3,4] x [b,h*w,4,1] = [b,h*w,3,1]
        vertice_world = torch.matmul(cam_ext.unsqueeze(1), vertice_homo[..., None]).squeeze(-1)  # [b,h*w,3]        
        # [b,1,3,3] x [b,h*w,3,1] = [b,h*w,3,1]        
        vertice_world_homo = self.lift_to_homo(vertice_world)
        persp = self.get_perspective_from_intrinsic(cam_int)  # [b,4,4]

        # [b,1,4,4] x [b,h*w,4,1] = [b,h*w,4,1]
        vertice_ndc = torch.matmul(persp.unsqueeze(1), vertice_world_homo[..., None]).squeeze(-1)  # [b,h*w,4]

        ############
        # render
        # 首次调用时创建 CUDA 光栅上下文，后续复用。
        if self.glctx is None:
            self.glctx = dr.RasterizeCudaContext(device=self.device)
        rast_out, rast_out_db = dr.rasterize(
            glctx=self.glctx, pos=vertice_ndc, tri=faces, resolution=(h, w), grad_db=True,
        )
        output, uv_da = dr.interpolate(
            attr=attributes, rast=rast_out, tri=faces, rast_db=rast_out_db, diff_attrs=[0, 1],
        )  # [b,h,w,4]
        output = output.permute(0, 3, 1, 2)
        return output, uv_da

    def construct_mesh(self, rgbd, cam_int):
        '''
        input:
            rgbd: [b,4,h,w]
                the first 3 channels for RGB
                the last channel for normalized disparity, in range [0,1]

            cam_int: [b,3,3] normalized camera intrinsic matrix
        
        output:
            mesh_dict: define mesh in camera space, includes the following keys
                vertice: [b,h*w,3]
                faces:  [nface,3]
                attributes: [b,h*w,c] include color and mask
        '''
        b, _, h, w = rgbd.size()
        
        ############
        # get pixel coordinates
        # 生成归一化屏幕坐标，再结合深度反投影到相机空间。
        pixel_2d = self.get_screen_pixel_coord(h, w)  # [1,h,w,2]
        pixel_2d_homo = self.lift_to_homo(pixel_2d)  # [1,h,w,3]

        ############
        # project pixels to 3D space
        rgbd = rgbd.permute(0, 2, 3, 1)  # [b,h,w,4]
        disparity = rgbd[..., -1:]  # [b,h,w,1]
        depth = torch.reciprocal(disparity + self.eps)  # [b,h,w,1]
        cam_int_inv = torch.inverse(cam_int)  # [b,3,3]
        # [b,1,1,3,3] x [1,h,w,3,1] = [b,h,w,3,1]
        pixel_3d = torch.matmul(cam_int_inv[:, None, None, :, :], pixel_2d_homo[..., None]).squeeze(-1)  # [b,h,w,3]
        pixel_3d = pixel_3d * depth  # [b,h,w,3]
        vertice = pixel_3d.reshape(b, h * w, 3)  # [b,h*w,3]

        ############
        # construct faces
        # 每个像素网格单元拆成两个三角形。
        faces = self.get_faces(h, w)  # [1,nface,3]
        faces = faces[0].int()  # [nface,3] nvdiffrast need int 32 format

        ############
        # compute attributes
        # attributes 最后一维放可见性 mask，便于后续筛除深度不连续区域。
        attr_color = rgbd[..., :-1].reshape(b, h * w, 3)  # [b,h*w,3]
        attr_mask = self.get_visible_mask(disparity).reshape(b, h * w, 1)  # [b,h*w,1]
        attr = torch.cat([attr_color, attr_mask], dim=-1)  # [b,h*w,4]

        mesh_dict = {
            "vertice": vertice,
            "faces": faces,
            "attributes": attr,
            "size": [h, w],
        }
        return mesh_dict

    def get_screen_pixel_coord(self, h, w):
        '''
        get normalized pixel coordinates on the screen
        x to left, y to down
        
        e.g.
        [0,0][1,0][2,0]
        [0,1][1,1][2,1]

        output:
            pixel_coord: [1,h,w,2]
        '''
        # 使用像素中心 (x+0.5, y+0.5) 进行采样。
        x = torch.arange(w).to(self.device)  # [w]
        y = torch.arange(h).to(self.device)  # [h]
        x = (x + 0.5) / w
        y = (y + 0.5) / h
        x = x[None, None, ..., None].repeat(1, h, 1, 1)  # [1,h,w,1]
        y = y[None, ..., None, None].repeat(1, 1, w, 1)  # [1,h,w,1]
        pixel_coord = torch.cat([x, y], dim=-1)  # [1,h,w,2]
        return pixel_coord
    
    def lift_to_homo(self, coord):
        '''
        return the homo version of coord

        input: coord [..., k]
        output: homo_coord [...,k+1]
        '''
        ones = torch.ones_like(coord[..., -1:])
        return torch.cat([coord, ones], dim=-1)

    def get_faces(self, h, w):
        '''
        get face connect information
        x to left, y to down

        e.g.
        [0,0][1,0][2,0]
        [0,1][1,1][2,1]

        faces: [1,nface,3]
        '''
        # 面片拓扑与图像网格一一对应。
        x = torch.arange(w - 1).to(self.device)  # [w-1]
        y = torch.arange(h - 1).to(self.device)  # [h-1]
        x = x[None, None, ..., None].repeat(1, h - 1, 1, 1)  # [1,h-1,w-1,1]
        y = y[None, ..., None, None].repeat(1, 1, w - 1, 1)  # [1,h-1,w-1,1]

        tl = y * w + x
        tr = y * w + x + 1
        bl = (y + 1) * w + x
        br = (y + 1) * w + x + 1

        faces_l = torch.cat([tl, bl, br], dim=-1).reshape(1, -1, 3)  # [1,(h-1)(w-1),3]
        faces_r = torch.cat([br, tr, tl], dim=-1).reshape(1, -1, 3)  # [1,(h-1)(w-1),3]

        return torch.cat([faces_l, faces_r], dim=1)  # [1,nface,3]

    def get_visible_mask(self, disparity, beta=10, alpha_threshold=0.3):
        '''
        filter the disparity map using sobel kernel, then mask out the edge (depth discontinuity)

        input:
            disparity: [b,h,w,1]
        
        output:
            vis_mask: [b,h,w,1]
        '''
        # Sobel 检测深度边缘；边缘处 alpha 低，mask 置 0。
        b, h, w, _ = disparity.size()
        disparity = disparity.reshape(b, 1, h, w)  # [b,1,h,w]
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().to(self.device)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).float().to(self.device)
        sobel_x = F.conv2d(disparity, kernel_x, padding=(1, 1))  # [b,1,h,w]
        sobel_y = F.conv2d(disparity, kernel_y, padding=(1, 1))  # [b,1,h,w]
        sobel_mag = torch.sqrt(sobel_x ** 2 + sobel_y ** 2).reshape(b, h, w, 1)  # [b,h,w,1]
        alpha = torch.exp(-1.0 * beta * sobel_mag)  # [b,h,w,1]
        vis_mask = torch.greater(alpha, alpha_threshold).float()
        return vis_mask

    def get_perspective_from_intrinsic(self, cam_int):
        '''
        input:
            cam_int: [b,3,3]
        
        output:
            persp: [b,4,4]
        '''
        # 由归一化内参构造 OpenGL 风格透视矩阵。
        fx, fy = cam_int[:, 0, 0], cam_int[:, 1, 1]  # [b]
        cx, cy = cam_int[:, 0, 2], cam_int[:, 1, 2]  # [b]

        one = torch.ones_like(cx)  # [b]
        zero = torch.zeros_like(cx)  # [b]

        near_z, far_z = self.near_z * one, self.far_z * one
        a = (near_z + far_z) / (far_z - near_z)
        b = -2.0 * near_z * far_z / (far_z - near_z)

        matrix = [[2.0 * fx, zero, 2.0 * cx - 1.0, zero],
                  [zero, 2.0 * fy, 2.0 * cy - 1.0, zero],
                  [zero, zero, a, b],
                  [zero, zero, one, zero]]
        # -> [[b,4],[b,4],[b,4],[b,4]] -> [b,4,4]        
        persp = torch.stack([torch.stack(row, dim=-1) for row in matrix], dim=-2)  # [b,4,4]
        return persp


#######################
# some helper functions
#######################
import cv2
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms


def image_to_tensor(img_path):
    rgb = transforms.ToTensor()(Image.open(img_path)).unsqueeze(0)
    return rgb


def disparity_to_tensor(disp_path):
    disp = cv2.imread(disp_path, -1) / (2 ** 16 - 1)
    disp = torch.from_numpy(disp)[None, None, ...]
    return disp.float()


if __name__ == "__main__":
    device = "cuda"
    render_save_path = "warp_rgbd/example/render_nvdiffrast.png"
    img_paths = ["warp_rgbd/example/rgb/0810.png", "warp_rgbd/example/rgb/0820.png"]
    disp_paths = ["warp_rgbd/example/depth/0810.png", "warp_rgbd/example/depth/0820.png"]
    h, w = 256, 384

    cam_int = torch.tensor([[0.58, 0, 0.5],
                            [0, 0.58, 0.5],
                            [0, 0, 1]]).to(device)
    cam_ext = torch.tensor([[1., 0., 0., 0.2],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 0.]]).to(device)

    bs = len(img_paths)
    cam_int = cam_int[None, ...].repeat(bs, 1, 1)  # [b,3,3]
    cam_ext = cam_ext[None, ...].repeat(bs, 1, 1)  # [b,3,3]

    rgbd = []
    for ip, dp in zip(img_paths, disp_paths):
        cur = torch.cat([image_to_tensor(ip), disparity_to_tensor(dp)], dim=1)
        cur = F.interpolate(cur, size=(h, w), mode="bilinear", align_corners=False)
        rgbd.append(cur)
    rgbd = torch.cat(rgbd, dim=0).to(device)  # [b,4,h,w]

    rgbd_renderer = RGBDRenderer(device)
    mesh = rgbd_renderer.construct_mesh(rgbd, cam_int)
    render = rgbd_renderer.render_mesh(mesh, cam_int, cam_ext)
    save_image(render, render_save_path)
