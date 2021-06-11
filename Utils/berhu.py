import torch
class Berhu(torch.nn.Module):
    def __init__(self,
        threshold:      float,
        adaptive:       bool=False,
        reduction:  str     = "none"
    ):

        super(Berhu, self).__init__()
        self.threshold = threshold
        self.adaptive = adaptive
        self.reduction = reduction
 

    def forward(self, 
        gt:         torch.Tensor,
        pred:       torch.Tensor,
        weights:    torch.Tensor=None,
    ) -> torch.Tensor:        
        L1 = torch.abs(gt - pred)
        if weights is not None:
            L1 = L1 * weights

        threshold = self.threshold if not self.adaptive else 0.2 * torch.max(error)
        berhu = torch.where(
            L1 <= threshold,
            L1,
            (L1 ** 2 + threshold ** 2) / (2.0 * threshold)
        )
        if self.reduction == "mean":
            if weights is None:
                return berhu.mean()
            else:
                return (berhu * weights).sum() / (weights.sum() + 1e-3)

        return berhu