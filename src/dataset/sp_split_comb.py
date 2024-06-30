import numpy as np
import torch
import torch.nn.functional as F

from .split_comb import SplitComb


class SPSplitComb(SplitComb):
    def split(self, data, vessel):
        splits = []
        vessel_splits = []
        nzhw = []
        z, h, w = data.shape

        nz = int(np.ceil(float(z) / self.side_len[0]))
        nh = int(np.ceil(float(h) / self.side_len[1]))
        nw = int(np.ceil(float(w) / self.side_len[2]))

        pad = (
            0,
            int(nw * self.side_len[2] + self.overlap[2] - w),
            0,
            int(nh * self.side_len[1] + self.overlap[1] - h),
            0,
            int(nz * self.side_len[0] + self.overlap[0] - z),
        )

        data = F.pad(data, pad, mode="constant", value=self.pad_value)
        vessel = F.pad(vessel, pad, mode="constant", value=self.pad_value)

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = int(iz * self.side_len[0])
                    ez = int((iz + 1) * self.side_len[0] + self.overlap[0])
                    sh = int(ih * self.side_len[1])
                    eh = int((ih + 1) * self.side_len[1] + self.overlap[1])
                    sw = int(iw * self.side_len[2])
                    ew = int((iw + 1) * self.side_len[2] + self.overlap[2])

                    vessel_split = data[None, sz:ez, sh:eh, sw:ew]

                    split = data[None, None, sz:ez, sh:eh, sw:ew]
                    splits.append(split)
                    vessel_splits.append(vessel_split)
                    nzhw.append((iz, ih, iw))

        return splits, vessel_splits, nzhw

    def combine(self, output, nzhw):
        idx = 0
        for idx, (iz, ih, iw) in enumerate(nzhw):
            sz = int(iz * self.side_len[0])
            sh = int(ih * self.side_len[1])
            sw = int(iw * self.side_len[2])
            # [N, 8]
            # 8-> id, prob, z_min, y_min, x_min, d, h, w
            output[idx, :, 2] += sz
            output[idx, :, 3] += sh
            output[idx, :, 4] += sw
        return output
