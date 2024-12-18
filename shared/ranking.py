from abc import ABC
from typing import Any

from pydantic import BaseModel

from shared.dataset import Dataset


class AbstractRanker(BaseModel, ABC):
    KEY: str

    def __init__(self, dataset: Dataset, **kwargs: dict[str, Any]):
        super().__init__(**kwargs)
        self.dataset = dataset
