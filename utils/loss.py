import mlx.core as mx
import mlx.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def __call__(self, input1, input2, y):
        diff = input1 - input2
        dist_sq = mx.sum(mx.multiply(diff, diff), 1)
        dist = mx.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = mx.maximum(mdist, 0.0)
        loss = y * dist_sq + (1 - y) * mx.multiply(dist, dist)
        loss = mx.sum(loss) / 2.0 / input1.shape[0]
        return loss


def threshold_contrastive_loss(input1, input2, margin):
    # dist < m: 1 (Same Class)
    # else: 0 (Different Class)
    diff = input1 - input2
    dist_sq = mx.sum(mx.multiply(diff, diff), 1)
    dist = mx.sqrt(dist_sq)
    return dist < margin, dist
