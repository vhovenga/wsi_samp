from .resnet import ResNetFeatureExtractor
from .clam_resnet import CLAMResNet50

FEATURE_EXTRACTORS = {
    "ResNetFeatureExtractor": ResNetFeatureExtractor,
    "CLAMResNet": CLAMResNet50
}