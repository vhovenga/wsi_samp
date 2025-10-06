from .basic_aggregators import MeanPoolAggregator, AttnMeanPoolAggregator, MaxPoolAggregator

AGGREGATORS = {
    "MeanPoolAggregator": MeanPoolAggregator,
    "AttnMeanPoolAggregator": AttnMeanPoolAggregator,
    "MaxPoolAggregator": MaxPoolAggregator,
}