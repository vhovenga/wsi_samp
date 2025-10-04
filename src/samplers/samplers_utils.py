def build_sampler(cfg):
    from . import SAMPLER_REGISTRY
    """Builds a sampler from the config."""
    name = cfg["name"]
    cls = SAMPLER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown sampler: {name}")
    params = cfg.get("params", {})
    return cls(**params)