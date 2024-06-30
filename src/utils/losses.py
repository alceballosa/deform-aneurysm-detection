import torch
from torch.nn import functional as F


def no_targets_cross_entropy_loss(class_logits, class_weight):
    """
    Cross entropy loss function for object detection classification
    where no target aneurysms are in the volume.

    Parameters
    __________
    class_logits : torch.Tensor
        Tensor of shape (N, C) where C is the number of classes.
    class_weight : torch.Tensor
        Tensor of shape (C, ) containing the class weights.

    Return
    ______
    loss : torch.Tensor
        Scalar tensor containing the loss.
    """
    # NOTE: we could potentially do negative mining here, as we
    # NOTE: have n_queries negative samples when entering this func.
    class_target = torch.zeros_like(class_logits)
    class_target[:, -1] = 1  # set only the background class to 1
    cross_entropy_loss = torch.nn.CrossEntropyLoss(
        weight=class_weight.to(class_logits.device), reduction="mean"
    )
    loss = cross_entropy_loss(class_logits, class_target)
    return loss


def focal_loss(classes_pred, targets, matched_indices_i, alpha, gamma):
    matched_classes_target = targets["labels"][matched_indices_i[1]].long()
    classes_target = torch.full(
        (classes_pred.shape[0],),
        classes_pred.shape[1] - 1,  # background class
        dtype=torch.int64,
        device=classes_pred.device,
    )
    classes_target[matched_indices_i[0]] = matched_classes_target
    classes_target = F.one_hot(classes_target, num_classes=classes_pred.shape[1])
    classes_prob = torch.sigmoid(classes_pred.detach())
    classes_prob = torch.clamp(classes_prob, 1e-4, 1.0 - 1e-4)
    alpha_factor = torch.ones_like(classes_prob) * alpha
    alpha_factor = torch.where(
        torch.eq(classes_target, 1), alpha_factor, 1.0 - alpha_factor
    )
    focal_weight = torch.where(
        torch.eq(classes_target, 1), 1.0 - classes_prob, classes_prob
    )
    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
    binary_cross_entropy = F.binary_cross_entropy_with_logits(
        classes_pred, classes_target.float(), reduction="none"
    )
    loss = (focal_weight * binary_cross_entropy).mean()
    return loss


def no_targets_focal_loss(classes_pred, alpha, gamma):
    classes_target = torch.full(
        (classes_pred.shape[0],),
        classes_pred.shape[1] - 1,  # background class
        dtype=torch.int64,
        device=classes_pred.device,
    )
    classes_target = F.one_hot(classes_target, num_classes=classes_pred.shape[1])
    classes_prob = torch.sigmoid(classes_pred.detach())
    classes_prob = torch.clamp(classes_prob, 1e-4, 1.0 - 1e-4)
    alpha_factor = torch.ones_like(classes_prob) * alpha
    alpha_factor = torch.where(
        torch.eq(classes_target, 1), alpha_factor, 1.0 - alpha_factor
    )
    focal_weight = torch.where(
        torch.eq(classes_target, 1), 1.0 - classes_prob, classes_prob
    )
    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
    binary_cross_entropy = F.binary_cross_entropy_with_logits(
        classes_pred, classes_target.float(), reduction="none"
    )
    loss = (focal_weight * binary_cross_entropy).mean()
    return loss


def bbox_iou_loss(box1, box2, DIoU=True, eps=1e-7):
    """
    Computes IoU loss given two sets of bounding boxes.
    """

    def zyxdhw2zyxzyx(box, dim=-1):
        ctr_zyx, dhw = torch.split(box, 3, dim)
        z1y1x1 = ctr_zyx - dhw / 2
        z2y2x2 = ctr_zyx + dhw / 2
        return torch.cat((z1y1x1, z2y2x2), dim)  # zyxzyx bbox

    box1 = zyxdhw2zyxzyx(box1)
    box2 = zyxdhw2zyxzyx(box2)
    # Get the coordinates of bounding boxes
    b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = box1.chunk(6, -1)
    b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = box2.chunk(6, -1)
    w1, h1, d1 = b1_x2 - b1_x1, b1_y2 - b1_y1, b1_z2 - b1_z1
    w2, h2, d2 = b2_x2 - b2_x1, b2_y2 - b2_y1, b2_z2 - b2_z1

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0) * (b1_z2.minimum(b2_z2) - b1_z1.maximum(b2_z1)).clamp(0) + eps

    # Union Area
    union = w1 * h1 * d1 + w2 * h2 * d2 - inter

    # IoU
    iou = inter / union
    if DIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(
            b2_x1
        )  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        cd = b1_z2.maximum(b2_z2) - b1_z1.minimum(b2_z1)  # convex depth
        c2 = cw**2 + ch**2 + cd**2 + eps  # convex diagonal squared
        rho2 = (
            (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
            + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            + +((b2_z1 + b2_z2 - b1_z1 - b1_z2) ** 2)
        ) / 4  # center dist ** 2
        loss = (1 - iou + rho2 / c2).mean()  # DIoU
        return loss
    loss = (1 - iou).mean()
    return loss  # IoU
