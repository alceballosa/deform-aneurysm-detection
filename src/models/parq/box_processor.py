import torch
import numpy as np

class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    Convertion between different representation of boxes (corners, dof, etc.)
    """

    def __init__(self, num_semcls, mean_size_path):
        self.num_semcls = num_semcls
        self.num_class = 1
        self.mean_size_path = mean_size_path
        if self.mean_size_path is not None:
            self.init_mean_size()

    def init_mean_size(self):
        self.type2class = {
            "aneurysm": 0,
        }

        self.class2type = {self.type2class[t]: t for t in self.type2class}

        self.typelong_mean_size = {}
        with open(self.mean_size_path, "r") as f:
            for line in f.readlines():
                type_cat, size = line.split(": ")
                size = size[1:-3].split(" ")
                size_ = []
                for j, s in enumerate(size):
                    if len(s) != 0:
                        size_.append(s)
                size = [float(size_[i]) for i in [0, 1, 2]]
                self.typelong_mean_size[type_cat] = size

        self.mean_size_arr = []
        self.type_mean_size = {}
        for i in range(self.num_class):
            object_type = self.class2type[i]
            for key, value in self.typelong_mean_size.items():
                key = key.split(",")
                if object_type in key:
                    self.mean_size_arr.append(value)
                    self.type_mean_size[object_type] = value
                    break

        self.mean_size_arr.append([1, 1, 1])
        self.type_mean_size["aneurysm"] = [1, 1, 1]
        self.mean_size_arr.append([1, 1, 1])
        self.type_mean_size["non-object"] = [1, 1, 1]
        self.mean_size_arr = torch.from_numpy(np.array(self.mean_size_arr))

    def compute_predicted_center(self, center_normalized):
        center_unnormalized = center_normalized * 1
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_scale, cls_prob):
        if self.mean_size_path is not None:
            pred_cls = cls_prob.argmax(-1)
            mean_size = self.mean_size_arr[pred_cls.data.cpu()]
            size_pred = torch.exp(size_scale) * mean_size.to(size_scale.device).float()
        else:
            return None
        return size_pred

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob, objectness_prob
