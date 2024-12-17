from abc import ABC, abstractmethod, abstractclassmethod

class Dataset(ABC):
    @abstractmethod
    def fetch_dataset(self):
        pass