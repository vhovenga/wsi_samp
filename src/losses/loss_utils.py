def build_loss(cfg):
    from . import LOSS_REGISTRY
    """
    Build a loss function from config.

    Example cfg:
      loss:
        name: CrossEntropyLoss
        params:
          label_smoothing: 0.1
          reduction: mean
    """
    name = cfg["name"]
    cls = LOSS_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown loss: {name}")
    params = cfg.get("params", {})
    return cls(**params)