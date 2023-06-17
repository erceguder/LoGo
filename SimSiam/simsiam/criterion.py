from torch import nn


class SimSiamLoss(nn.Module):
    def asymmetric_loss(self, p, z):
        z = z.detach()  # stop gradient
        return - nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, z1, z2, p1, p2):
        loss1 = self.asymmetric_loss(p1, z2)
        loss2 = self.asymmetric_loss(p2, z1)

        return 0.5 * loss1 + 0.5 * loss2