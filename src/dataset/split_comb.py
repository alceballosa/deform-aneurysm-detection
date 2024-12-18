import numpy as np
import torch
import torch.nn.functional as F


class SplitComb:
    def __init__(
        self,
        crop_size: list = [64, 128, 128],
        overlap: list = [16, 32, 32],
        pad_value: float = 0,
    ):
        self.side_len = [
            crop_size[0] - overlap[0],
            crop_size[1] - overlap[1],
            crop_size[2] - overlap[2],
        ]
        self.overlap = overlap
        self.pad_value = pad_value

    def split(self, data: torch.Tensor):
        splits = []
        splits_boxes = []
        z, h, w = data.size()

        nz = int(np.ceil(float(z) / self.side_len[0]))
        nh = int(np.ceil(float(h) / self.side_len[1]))
        nw = int(np.ceil(float(w) / self.side_len[2]))

        nzhw = [nz, nh, nw]
        pad = (
            0,
            int(nw * self.side_len[2] + self.overlap[2] - w),
            0,
            int(nh * self.side_len[1] + self.overlap[1] - h),
            0,
            int(nz * self.side_len[0] + self.overlap[0] - z),
        )

        data = F.pad(data, pad, mode="constant", value=self.pad_value)

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = int(iz * self.side_len[0])
                    ez = int((iz + 1) * self.side_len[0] + self.overlap[0])
                    sh = int(ih * self.side_len[1])
                    eh = int((ih + 1) * self.side_len[1] + self.overlap[1])
                    sw = int(iw * self.side_len[2])
                    ew = int((iw + 1) * self.side_len[2] + self.overlap[2])
                    box = [sz, ez, sh, eh, sw, ew]
                    # use numpy to avoid issues with torch.cat being slow on cpu
                    split = data[None, None, sz:ez, sh:eh, sw:ew].numpy()
                    splits.append(split)
                    splits_boxes.append(box)

        # splits = np.concatenate(splits, 0)
        return splits, nzhw, splits_boxes

    def combine(self, output, nzhw=None):
        """
        outputs: (B,N,8)
        """
        assert nzhw is not None
        nz, nh, nw = nzhw
        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = int(iz * self.side_len[0])
                    sh = int(ih * self.side_len[1])
                    sw = int(iw * self.side_len[2])
                    # [N, 8]
                    # 8-> id, prob, z_min, y_min, x_min, d, h, w
                    output[idx, :, 2] += sz
                    output[idx, :, 3] += sh
                    output[idx, :, 4] += sw
                    idx += 1
        return output
