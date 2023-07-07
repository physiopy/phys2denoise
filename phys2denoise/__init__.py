from ._version import get_versions
from .metrics import __all__, cardiac, chest_belt, gas, multimodal, responses, utils

__version__ = get_versions()["version"]
del get_versions
