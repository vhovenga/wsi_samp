from torchmetrics import MetricCollection

def build_metrics(cfg, prefix=""):
    from . import METRIC_REGISTRY
    """Builds a metric collection from the config."""
    metrics_dict = {}
    for name, kwargs in cfg.items():
        cls = METRIC_REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"Unknown metric: {name}")

        metrics_dict[name] = cls(**kwargs)
    return MetricCollection(metrics_dict, prefix=prefix)