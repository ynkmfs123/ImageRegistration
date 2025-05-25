import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
import time
import os
from PIL import Image
from modules.tps import pytorch as TPS

class REG:

    def __init__(self, model=None,
                 dev=torch.device('cpu'),
                 fixed_tps=False):
        self.dev = dev
        if model is None:
            abs_path = os.path.dirname(os.path.abspath(__file__))
            model = abs_path + '/reg.pth'
            backbone_nfeats = 64
            mode = 'reg'
        self.net = Model(enc_channels=[1, 32, 64, backbone_nfeats], fixed_tps=fixed_tps, mode=mode).to(dev)
        self.net.load_state_dict(torch.load(model, map_location=dev))
        self.net.eval().to(dev)

    def detectAndCompute(self, data, mask=None, top_k=2048, return_map=False, threshold=25.):
        t0 = time.time()
        scales = [1]

        kpts_list, descs_list, scores_list = [], [], []
        hd_map = None

        for scale in scales:
            with torch.no_grad():
                data_min = data.min()
                data_max = data.max()
                data_norm = 255 * (data - data_min) / (data_max - data_min)
                data_norm = data_norm.astype(np.uint8)
                img = Image.fromarray(data_norm, mode='L')
                img_array = np.array(img)
                if img_array.ndim == 2:
                    og_img = np.stack([np.array(img)] * 3, axis=-1)
                else:
                    og_img = img
                img = torch.tensor(og_img, dtype=torch.float32, device=self.dev).permute(2, 0, 1).unsqueeze(0) / 255.
                kpts, descs, fmap = self.net(img, NMS=True, threshold=threshold, return_tensors=True, top_k=top_k)

                score_map = fmap['map'][0].squeeze(0).cpu().numpy()

                kpts, descs = kpts[0]['xy'].cpu().numpy().astype(np.int16), descs[0].cpu().numpy()
                scores = score_map[kpts[:, 1], kpts[:, 0]]
                scores /= score_map.max()
                sort_idx = np.argsort(-scores)
                kpts, descs, scores = kpts[sort_idx], descs[sort_idx], scores[sort_idx]
                if return_map and hd_map is None:
                    max_val = float(score_map.max())
                    for p in kpts.astype(np.int32):
                        if False:  # score_map[p[1],p[0]] > threshold:
                            cv2.drawMarker(score_map, (p[0], p[1]), max_val, cv2.MARKER_CROSS, 6, 2)
                    hd_map = score_map
                kpts = kpts / scale
                kpts_list.append(kpts)
                descs_list.append(descs)
                scores_list.append(scores)

        if len(scales) > 1:
            perscale_kpts = top_k // len(scales)
            all_kpts = np.vstack([kpts[:perscale_kpts] for kpts in kpts_list[:-1]])
            all_descs = np.vstack([descs[:perscale_kpts] for descs in descs_list[:-1]])
            all_scores = np.hstack([scores[:perscale_kpts] for scores in scores_list[:-1]])
            all_kpts = np.vstack([all_kpts, kpts_list[-1][:(top_k - len(all_kpts))]])
            all_descs = np.vstack([all_descs, descs_list[-1][:(top_k - len(all_descs))]])
            all_scores = np.hstack([all_scores, scores_list[-1][:(top_k - len(all_scores))]])
        else:
            all_kpts = kpts_list[0];
            all_descs = descs_list[0];
            all_scores = scores_list[0]
        cv_kps = [cv2.KeyPoint(all_kpts[i][0], all_kpts[i][1], 6, 0, all_scores[i]) for i in range(len(all_kpts))]
        if return_map:
            return cv_kps, all_descs, hd_map
        else:
            return cv_kps, all_descs

    def detect(self, img, _=None):
        return self.detectAndCompute(img)[0]


class InterpolateSparse2d(nn.Module):
    def __init__(self, mode='bicubic'):
        super().__init__()
        self.mode = mode

    def normgrid(self, x, H, W):
        return 2. * (x / (torch.tensor([W - 1, H - 1], device=x.device, dtype=x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        grid = self.normgrid(pos, H, W).unsqueeze(0).unsqueeze(-2)
        x = F.grid_sample(x.unsqueeze(0), grid, mode=self.mode, align_corners=True)
        return x.permute(0, 2, 3, 1).squeeze(0).squeeze(-2)


class KeypointSampler(nn.Module):
    def __init__(self, window_size=8):
        super().__init__()
        self.window_size = window_size

    def gridify(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.window_size, self.window_size) \
            .unfold(3, self.window_size, self.window_size) \
            .reshape(B, C, H // self.window_size, W // self.window_size, self.window_size ** 2)
        return x

    def sample(self, grid):
        chooser = torch.distributions.Categorical(logits=grid)
        choices = chooser.sample()
        selected_choices = torch.gather(grid, -1, choices.unsqueeze(-1)).squeeze(-1)
        flipper = torch.distributions.Bernoulli(logits=selected_choices)
        accepted_choices = flipper.sample()
        log_probs = chooser.log_prob(choices) + flipper.log_prob(accepted_choices)
        accept_mask = accepted_choices.gt(0)

        return log_probs.squeeze(1), choices, accept_mask.squeeze(1)

    def forward(self, x):
        B, C, H, W = x.shape
        keypoint_cells = self.gridify(x)
        idx_cells = self.gridify(torch.dstack(torch.meshgrid(torch.arange(x.shape[-2], dtype=torch.float32),
                                                             torch.arange(x.shape[-1], dtype=torch.float32),
                                                             )) \
                                 .permute(2, 0, 1).unsqueeze(0)
                                 .expand(B, -1, -1, -1)).to(x.device)
        log_probs, idx, mask = self.sample(keypoint_cells)
        keypoints = torch.gather(idx_cells, -1, idx.repeat(1, 2, 1, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 3, 1)
        xy_probs = [{'xy': keypoints[b][mask[b]].flip(-1), 'logprobs': log_probs[b][mask[b]]}
                    for b in range(B)]

        return xy_probs


class Matcher(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, T=1.):
        Dmat = 2. - torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
        logprob_rows = F.log_softmax(Dmat * T, dim=1)
        logprob_cols = F.log_softmax(Dmat.t() * T, dim=1)
        choice_rows = torch.argmax(logprob_rows, dim=1)
        choice_cols = torch.argmax(logprob_cols, dim=1)
        seq = torch.arange(choice_cols.shape[0], dtype=choice_cols.dtype, device=choice_cols.device)
        mutual = choice_rows[choice_cols] == seq

        logprob_rows = torch.gather(logprob_rows, -1, choice_rows.unsqueeze(-1)).squeeze(-1)
        logprob_cols = torch.gather(logprob_cols, -1, choice_cols.unsqueeze(-1)).squeeze(-1)

        log_probs = logprob_rows[choice_cols[mutual]] + logprob_cols[seq[mutual]]
        dmatches = torch.cat((choice_cols[mutual].unsqueeze(-1), seq[mutual].unsqueeze(-1)), dim=1)

        return log_probs, dmatches


class DenseMatcher(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, T=1.):
        Dmat = 2. - torch.cdist(x, y)
        logprob_rows = F.log_softmax(Dmat * T, dim=1)
        logprob_cols = F.log_softmax(Dmat * T, dim=0)

        return logprob_rows + logprob_cols


class Model(nn.Module):
    def __init__(self, enc_channels=[1, 32, 64, 128], fixed_tps=False, mode=None):
        super().__init__()
        self.net = UNet(enc_channels)
        self.detector = KeypointSampler()
        self.interpolator = InterpolateSparse2d()
        hn_out_ch = 64
        self.tps_net = ThinPlateNet(in_channels=enc_channels[-1], nchannels=enc_channels[0],
                                    fixed_tps=fixed_tps)
        self.hardnet = HardNet(nchannels=enc_channels[0], out_ch=hn_out_ch)

        self.nchannels = enc_channels[0]
        self.enc_channels = enc_channels
        self.mode = mode
        self.fusion_layer = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                                          nn.Linear(128, 128), nn.Sigmoid())

    def subpix_refine(self, score_map, xy, size=3):
        from kornia.geometry.subpix import dsnt
        if size % 2 == 0:
            raise RuntimeError('Grid size must be odd')
        _, H, W = score_map.shape
        score_map = score_map.unsqueeze(1).expand(xy.shape[0], -1, -1, -1)
        g = torch.arange(size) - size // 2
        gy, gx = torch.meshgrid(g, g)
        center_grid = torch.cat([gx.unsqueeze(-1), gy.unsqueeze(-1)], -1).to(xy.device)
        grids = center_grid.unsqueeze(0).repeat(xy.shape[0], 1, 1, 1)
        grids = (grids + xy.view(-1, 1, 1, 2)) / torch.tensor([W - 1, H - 1]).to(xy.device)
        grids = grids * 2 - 1
        patches_scores = F.grid_sample(score_map, grids, mode='nearest', align_corners=True)
        patches_scores = F.softmax(patches_scores.view(-1, size * size) / 1., dim=-1).view(-1, 1, size, size)
        xy_offsets = dsnt.spatial_expectation2d(patches_scores, False).view(-1, 2) - size // 2
        xy = xy.float() + xy_offsets
        return xy

    def NMS(self, x, threshold=3., kernel_size=3):
        pad = kernel_size // 2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        return pos.nonzero()[..., 1:].flip(-1)

    def sample_descs(self, feature_map, kpts, H, W):  # 从特征图中根据给定的关键点坐标采样关键点的描述符，并返回这些描述符。
        return self.interpolator(feature_map, kpts, H, W).contiguous()

    def forward(self, x, NMS=False, threshold=3., return_tensors=False, top_k=None):
        if self.nchannels == 1 and x.shape[1] != 1:
            x = torch.mean(x, axis=1, keepdim=True)
        B, C, H, W = x.shape
        out = self.net(x)
        kpts = [{'xy': self.NMS(out['map'][b], threshold)} for b in range(B)]

        if top_k is not None:
            for b in range(B):
                scores = out['map'][b].squeeze(0)[kpts[b]['xy'][:, 1].long(), kpts[b]['xy'][:, 0].long()]
                sorted_idx = torch.argsort(-scores)
                kpts[b]['xy'] = kpts[b]['xy'][sorted_idx[:top_k]]
                if 'logprobs' in kpts[b]:
                    kpts[b]['logprobs'] = kpts[b]['xy'][sorted_idx[:top_k]]

        patches = self.tps_net(out['feat'], x, kpts, H, W)
        for b in range(B):
            kpts[b]['patches'] = patches[b]

        if NMS:
            if len(kpts[b]['xy']) == 1:
                raise RuntimeError('No keypoints detected.')
            final_desc = torch.cat((
                self.hardnet(patches[b]),
                self.interpolator(out['feat'][b], kpts[b]['xy'], H, W)
            ), dim=1)
            final_desc = self.fusion_layer(final_desc) * final_desc
            descs = [F.normalize(final_desc) for b in range(B)]
        for b, k in enumerate(kpts):
            k['xy'] = self.subpix_refine(out['map'][b], k['xy'])
        if not return_tensors:
            return kpts, descs
        else:
            return kpts, descs, out


class ThinPlateNet(nn.Module):
    def __init__(self, in_channels, nchannels=1, ctrlpts=(8, 8), fixed_tps=False):
        super().__init__()
        self.ctrlpts = ctrlpts
        self.nctrl = ctrlpts[0] * ctrlpts[1]
        self.nparam = (self.nctrl + 2)
        self.interpolator = InterpolateSparse2d(mode='bilinear')
        self.fixed_tps = fixed_tps
        self.fcn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels * 2, affine=False),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels * 2, 3, bias=False),
            nn.BatchNorm2d(in_channels * 2, affine=False),
            nn.ReLU(),
        )
        self.attn = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels * 4),
            nn.BatchNorm1d(in_channels * 4, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels * 4, in_channels * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels * 4, self.nparam * 2),
            nn.Tanh(),
        )
        for i in [-2, -5, -9]:
            self.attn[i].weight.data.normal_(0., 1e-5)
            self.attn[i].bias.data.zero_()

    def get_polar_grid(self, keypts, Hs, Ws, coords='linear', gridSize=(32, 32), maxR=32.):
        maxR = torch.ones_like(keypts[:, 0]) * maxR
        self.batchSize = keypts.shape[0]

        ident = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device=keypts.device).expand(self.batchSize, -1, -1)
        grid = F.affine_grid(ident, (self.batchSize, 1) + gridSize, align_corners=False)
        # print('nn', grid.shape)
        grid_y = grid[..., 0].view(self.batchSize, -1)
        grid_x = grid[..., 1].view(self.batchSize, -1)

        maxR = torch.unsqueeze(maxR, -1).expand(-1, grid_y.shape[-1]).float().to(keypts.device)
        normGrid = (grid_y + 1) / 2
        if coords == "log":
            r_s_ = torch.exp(normGrid * torch.log(maxR))
        elif coords == "linear":
            r_s_ = 1 + normGrid * (maxR - 1)
        else:
            raise RuntimeError('Invalid coords type, choose [log, linear]')
        r_s = (r_s_ - 1) / (maxR - 1) * 2 * maxR / Ws
        t_s = (
                      grid_x + 1
              ) * np.pi

        x_coord = torch.unsqueeze(keypts[:, 0], -1).expand(-1, grid_x.shape[-1]) / Ws * 2. - 1.
        y_coord = torch.unsqueeze(keypts[:, 1], -1).expand(-1, grid_y.shape[-1]) / Hs * 2. - 1.

        aspectRatio = Ws / Hs
        x_s = r_s * torch.cos(
            t_s
        ) + x_coord
        y_s = r_s * torch.sin(
            t_s
        ) * aspectRatio + y_coord
        polargrid = torch.cat(
            (x_s.view(self.batchSize, gridSize[0], gridSize[1], 1),
             y_s.view(self.batchSize, gridSize[0], gridSize[1], 1)),
            -1)
        return polargrid

    def forward(self, x, in_imgs, keypts, Ho, Wo):
        patches = []
        B, C, _, _ = x.shape
        Theta = self.fcn(x)
        for b in range(B):
            if keypts[b]['xy'] is not None and len(keypts[b]['xy']) >= 16:
                polargrid = self.get_polar_grid(keypts[b]['xy'], Ho, Wo)
                N, H, W, _ = polargrid.shape

                kfactor = 0.3
                offset = (1.0 - kfactor) / 2.
                vmin = polargrid.view(N, -1, 2).min(1)[0].unsqueeze(1).unsqueeze(1)
                vmax = polargrid.view(N, -1, 2).max(1)[0].unsqueeze(1).unsqueeze(1)
                ptp = vmax - vmin
                polargrid = (polargrid - vmin) / ptp
                polargrid = polargrid * kfactor + offset
                grid_img = polargrid.permute(0, 3, 1, 2)
                ctrl = F.interpolate(grid_img, self.ctrlpts).permute(0, 2, 3, 1).view(N, -1, 2)
                theta = self.interpolator(Theta[b], keypts[b]['xy'], Ho, Wo)
                theta = self.attn(theta)
                theta = theta.view(-1, self.nparam, 2)

                I_polargrid = theta.new(N, H, W, 3)
                I_polargrid[..., 0] = 1.0
                I_polargrid[..., 1:] = polargrid
                if not self.fixed_tps:
                    z = TPS.tps(theta, ctrl, I_polargrid)
                    tps_warper = (I_polargrid[..., 1:] + z)
                else:
                    tps_warper = polargrid
                tps_warper = (tps_warper - offset) / kfactor
                tps_warper = tps_warper * ptp + vmin
                curr_patches = F.grid_sample(in_imgs[b].expand(N, -1, -1, -1),
                                             tps_warper, align_corners=False, padding_mode='zeros')
                patches.append(curr_patches)
            else:
                patches.append(None)
        return patches


class Pad2D(torch.nn.Module):
    def __init__(self, pad, mode):
        super().__init__()
        self.pad = pad
        self.mode = mode

    def forward(self, x):
        return F.pad(x, pad=self.pad, mode=self.mode)


class HardNet(nn.Module):
    def __init__(self, nchannels=3, out_ch=128):
        super().__init__()

        self.nchannels = nchannels

        self.features = nn.Sequential(
            nn.InstanceNorm2d(self.nchannels),
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(self.nchannels, 32, 3, bias=False, padding=(0, 1)),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(32, 32, 3, bias=False, padding=(0, 1)),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(32, 64, 3, bias=False, padding=(0, 1), stride=2),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(64, 64, 3, bias=False, padding=(0, 1)),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(64, 64, 3, bias=False, padding=(0, 1), stride=2),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(64, 64, 3, bias=False, padding=(0, 1)),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.AvgPool2d((8, 1), stride=1),
            nn.Conv2d(64, out_ch, (1, 3), bias=False),
            nn.BatchNorm2d(out_ch, affine=False),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, (1, 3), bias=False),
            nn.BatchNorm2d(out_ch, affine=False),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, (1, 3), bias=False),
            nn.BatchNorm2d(out_ch, affine=False),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, (1, 2), bias=False),
            nn.BatchNorm2d(out_ch, affine=False)
        )

    def forward(self, x):
        if x is not None:
            x = self.features(x).squeeze(-1).squeeze(-1)
        return x


class SmallFCN(nn.Module):
    def __init__(self, nchannels=3):
        super().__init__()

        self.nchannels = nchannels

        self.features = nn.Sequential(
            nn.InstanceNorm2d(self.nchannels),
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(self.nchannels, 8, 3, bias=False, padding=(0, 1)),
            nn.BatchNorm2d(8, affine=False),
            nn.ReLU(),
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(8, 16, 3, bias=False, padding=(0, 1), stride=2),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            Pad2D(pad=(0, 0, 1, 1), mode='circular'),
            nn.Conv2d(16, 16, 3, bias=False, padding=(0, 1), stride=2),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.AvgPool2d((8, 1), stride=1),
            nn.Conv2d(16, 32, (1, 8), bias=False),
            nn.BatchNorm2d(32, affine=False),
        )

    def forward(self, x):
        x = self.features(x).squeeze(-1).squeeze(-1)
        x = F.normalize(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.Sequential(
            # Gaussian2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 3, bias=False, padding=1),
            nn.BatchNorm2d(out_ch, affine=False),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, bias=False, padding=1),
            nn.BatchNorm2d(out_ch, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.convs(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=0, bias=False),
            nn.BatchNorm2d(out_ch, affine=False),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=0, bias=False),
            nn.BatchNorm2d(out_ch, affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.InstanceNorm2d(channels[0], affine=False)
        self.blocks = nn.ModuleList([DownBlock(channels[i], channels[i + 1])
                                     for i in range(len(channels) - 1)])

    def forward(self, x):
        x = self.norm(x)
        features = [x]

        for b in self.blocks:
            x = b(x)
            features.append(x)
        return features


class Decoder(nn.Module):
    def __init__(self, enc_ch, dec_ch):
        super().__init__()
        enc_ch = enc_ch[::-1]
        self.convs = nn.ModuleList([UpBlock(enc_ch[i + 1] + dec_ch[i], dec_ch[i + 1])
                                    for i in range(len(dec_ch) - 2)])
        self.conv_heatmap = nn.Sequential(
            nn.Conv2d(dec_ch[-2], dec_ch[-2], 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch[-2], affine=False),
            nn.ReLU(),
            nn.Conv2d(dec_ch[-2], 1, 1),
        )

    def forward(self, x):
        x = x[::-1]
        x_next = x[0]
        for i in range(len(self.convs)):
            upsampled = F.interpolate(x_next, size=x[i + 1].size()[-2:], mode='bilinear', align_corners=True)
            x_next = torch.cat([upsampled, x[i + 1]], dim=1)
            x_next = self.convs[i](x_next)
        x_next = F.interpolate(x_next, size=x[-1].size()[-2:], mode='bicubic', align_corners=True)
        return self.conv_heatmap(x_next)


class UNet(nn.Module):

    def __init__(self, enc_channels=[1, 32, 64, 128]):
        super().__init__()
        dec_channels = enc_channels[::-1]
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(enc_channels, dec_channels)
        self.nchannels = enc_channels[0]
        self.features = nn.Sequential(
            nn.Conv2d(enc_channels[-1], enc_channels[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[-1], affine=False),
            nn.ReLU(),
            nn.Conv2d(enc_channels[-1], enc_channels[-1], 1),
            nn.BatchNorm2d(enc_channels[-1], affine=False)
        )

    def forward(self, x):
        if self.nchannels == 1 and x.shape[1] != 1:
            x = torch.mean(x, axis=1, keepdim=True)
        feats = self.encoder(x)
        out = self.decoder(feats)
        feat = self.features(feats[-1])
        return {'map': out, 'feat': feat}


