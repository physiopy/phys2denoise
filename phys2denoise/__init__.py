from . import _version
from .metrics import __all__, cardiac, chest_belt, gas, multimodal, responses, utils

__version__ = _version.get_versions()["version"]
