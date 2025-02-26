from abc import ABC, abstractmethod

class BaseProvider(ABC):
    def __init__(self, collection_name: str, uid_field: str):
        self.collection_name = collection_name
        self.uid_field = uid_field
        
    @abstractmethod
    def get_schema(self):
        pass

    @abstractmethod
    def parse_data(self, json_data):
        pass