from abc import ABC, abstractmethod

class BaseProvider(ABC):
    @abstractmethod
    def get_schema(self):
        pass

    @abstractmethod
    def parse_data(self, json_data):
        pass