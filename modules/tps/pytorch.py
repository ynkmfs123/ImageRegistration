# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

import torch

def tps(theta, ctrl, grid):
    '''Evaluate the thin-plate-spline (TPS) surface at xy locations arranged in a grid.
    The TPS surface is a minimum bend interpolation surface defined by a set of control points.
    The function value for a x,y location is given by
    
        TPS(x,y) := theta[-3] + theta[-2]*x + theta[-1]*y + \sum_t=0,T theta[t] U(x,y,ctrl[t])
        
    This method computes the TPS value for multiple batches over multiple grid locations for 2 
    surfaces in one go.
    
    Params
    ------
    theta: Nx(T+3)x2 tensor, or Nx(T+2)x2 tensor
        Batch size N, T+3 or T+2 (reduced form) model parameters for T control points in dx and dy.
    ctrl: NxTx2 tensor or Tx2 tensor
        T control points in normalized image coordinates [0..1]
    grid: NxHxWx3 tensor
        Grid locations to evaluate with homogeneous 1 in first coordinate.
        
    Returns
    -------
    z: NxHxWx2 tensor
        Function values at each grid location in dx and dy.
    '''
    '''
    这是一个用于在网格中的各个xy位置评估薄板样条（Thin-Plate-Spline，TPS）曲面的函数的Python文档字符串。以下是对这个文档字符串的关键部分的解释：
    theta：这个参数表示TPS曲面的模型参数,形状为Nx(T+3)x2或Nx(T+2)x2的张量。其中，“N”是批量大小，“T”是控制点的数量，每个控制点都有两个参数（dx和dy）
    ctrl：这个参数表示在标准化图像坐标[0..1]中的控制点。它是一个形状为NxTx2或Tx2的张量，其中“N”是批量大小，“T”是控制点的数量，每个控制点由两个坐标（x和y）表示
    grid：这个参数表示您希望评估TPS曲面的网格位置。它是一个形状为NxHxWx3的张量，其中“N”是批量大小，“H”是网格的高度，“W”是网格的宽度，最后一个维度有三个元素，第一个元素是齐次坐标1。
    z：这是函数的输出，表示每个网格位置的函数值，以dx和dy表示。它是一个形状为NxHxWx2的张量，其中“N”是批量大小，“H”是网格的高度，“W”是网格的宽度，每个位置有两个值（dx和dy）。
    该函数本身根据提供的模型参数和控制点在指定的网格位置上计算TPS函数值。它使用文档字符串中提到的TPS公式。
    '''
    
    N, H, W, _ = grid.size()#获取一个四维张量grid的尺寸信息 2048,32,32 对应I_polargrid

    if ctrl.dim() == 2:#检查变量ctrl的维度是否为2，即是否是一个二维张量
        ctrl = ctrl.expand(N, *ctrl.size())#被扩展（expanded）为一个与N（批处理大小）相同大小的三维张量。例如形状为(T, 2)，那么将被扩展为三维张量，形状为(N,T,2)，N是批处理大小，T是控制点数量，2表示每个控制点的坐标。
    
    T = ctrl.shape[1]#ctrl张量中控制点的数量 64
    #print('T',T)
    
    diff = grid[...,1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1)#计算每个网格位置与控制点之间的差异（差矢量）
    D = torch.sqrt((diff**2).sum(-1))#计算了diff张量中每个差异向量的模长（欧几里得距离）
    U = (D**2) * torch.log(D + 1e-6)#计算了U张量，它是D的函数
    #这个计算U的目的是为了生成薄板样条插值中的权重，其中距离越远的控制点对某个网格位置的影响越小，而距离越近的控制点对其影响越大。函数 log(D + 1e-6) 用于平滑权重，以确保 D 接近 0 时不会出现无限大的值。
    #-----------
    w, a = theta[:, :-3, :], theta[:, -3:, :]
    # w, a = theta[:, :-2, :], theta[:, -2:, :]
    # a = torch.cat((torch.zeros_like(a[:, :1, :], requires_grad=False), a), dim = 1)
    

    reduced = T + 2  == theta.shape[1]
    ############
    if reduced:
        w = torch.cat((-w.sum(dim=1, keepdim=True), w), dim=1) 

    # U is NxHxWxT
    b = torch.bmm(U.view(N, -1, T), w).view(N,H,W,2)
    # b is NxHxWx2
    z = torch.bmm(grid.view(N,-1,3), a).view(N,H,W,2) + b
    
    return z

def tps_grid(theta, ctrl, size):
    '''Compute a thin-plate-spline grid from parameters for sampling.
    
    Params
    ------
    theta: Nx(T+3)x2 tensor
        Batch size N, T+3 model parameters for T control points in dx and dy.
    ctrl: NxTx2 tensor, or Tx2 tensor
        T control points in normalized image coordinates [0..1]
    size: tuple
        Output grid size as NxCxHxW. C unused. This defines the output image
        size when sampling.
    
    Returns
    -------
    grid : NxHxWx2 tensor
        Grid suitable for sampling in pytorch containing source image
        locations for each output pixel.
    '''    
    N, _, H, W = size#N是批量大小，H是图像高度，W是图像宽度

    grid = theta.new(N, H, W, 3)#创建了一个新的张量 grid，其形状为 (N, H, W, 3)，其中 N 是批量大小，H 是图像高度，W 是图像宽度，3 表示三个通道。
    grid[:, :, :, 0] = 1.# 将 grid 张量的第一个通道（通道索引为0）设置为1，这是一个恒定通道。
    grid[:, :, :, 1] = torch.linspace(0, 1, W)#在 grid 张量的第二个通道中生成从0到1的等间隔值，数量为图像的宽度 W
    #在 grid 张量的第三个通道中生成从0到1的等间隔值，数量为图像的高度 H。unsqueeze(-1) 将结果的维度从 (N, H, W) 扩展为 (N, H, W, 1)，以便与后续计算进行广播
    grid[:, :, :, 2] = torch.linspace(0, 1, H).unsqueeze(-1)   
    
    z = tps(theta, ctrl, grid)
    return (grid[...,1:] + z)*2-1 # [-1,1] range required by F.sample_grid

def tps_sparse(theta, ctrl, xy):
    if xy.dim() == 2:
        xy = xy.expand(theta.shape[0], *xy.size())

    N, M = xy.shape[:2]
    grid = xy.new(N, M, 3)
    grid[..., 0] = 1.
    grid[..., 1:] = xy

    z = tps(theta, ctrl, grid.view(N,M,1,3))
    return xy + z.view(N, M, 2)

def uniform_grid(shape):
    '''Uniformly places control points aranged in grid accross normalized image coordinates.
    
    Params
    ------
    shape : tuple
        HxW defining the number of control points in height and width dimension

    Returns
    -------
    points: HxWx2 tensor
        Control points over [0,1] normalized image range.
    '''
    H,W = shape[:2]    
    c = torch.zeros(H, W, 2)
    c[..., 0] = torch.linspace(0, 1, W)
    c[..., 1] = torch.linspace(0, 1, H).unsqueeze(-1)
    return c

if __name__ == '__main__':
    c = torch.tensor([
        [0., 0],
        [1., 0],
        [1., 1],
        [0, 1],
    ]).unsqueeze(0)
    theta = torch.zeros(1, 4+3, 2)
    size= (1,1,6,3)
    print(tps_grid(theta, c, size).shape)