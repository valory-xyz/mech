from typing import Dict

from packages.valory.skills.abstract_round_abci.io_.load import Loader
from packages.valory.skills.abstract_round_abci.io_.store import SupportedObjectType


class NaiveLoader(Loader):
    """A simple loader that doesn't deserialize the response from ipfs, but returns it as is."""

    def load(self, serialized_objects: Dict[str, str]) -> SupportedObjectType:
        """
        Return the objects as is.

        :param serialized_objects: A mapping of filenames to serialized object they contained.
        :return: the loaded file(s).
        """
        return serialized_objects
