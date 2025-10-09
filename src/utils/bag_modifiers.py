import torch 

class BagEMABuffer:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.buf = {}

    @torch.no_grad()
    def update(self, bag_ids, Z_list):
        for bid, z in zip(bag_ids, Z_list):
            z = z.detach()
            if bid not in self.buf:
                self.buf[bid] = z.clone()
            else:
                self.buf[bid].mul_(1 - self.alpha).add_(z, alpha=self.alpha)

    @torch.no_grad()
    def get(self, bag_ids):
        return [self.buf.get(bid) for bid in bag_ids]

    @torch.no_grad()
    def compute(self, bag_ids, Z_list):
        """Returns EMA-smoothed embeddings; updates buffer in-place."""
        self.update(bag_ids, Z_list)
        mu_list = self.get(bag_ids)
        mu_list = [z if mu is None else mu for z, mu in zip(Z_list, mu_list)]
        return torch.stack(mu_list, dim=0)


BAG_MODIFIERS = {
    "BagEMABuffer": BagEMABuffer,
}

def build_bag_modifier(cfg):
    name = cfg["name"]
    cls = BAG_MODIFIERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown bag modifier: {name}")
    params = cfg.get("params", {})
    return cls(**params)

