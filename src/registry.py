import importlib
import os
from pathlib import Path

class ProviderRegistry:
    def __init__(self):
        self.providers = {}

    def load_providers(self):
        providers_path = Path(__file__).parent / "providers"
        for file in providers_path.glob("*_provider.py"):
            module_name = f"src.providers.{file.stem}"
            module = importlib.import_module(module_name)
            if hasattr(module, "__all__") and file.stem != "base_provider":  # base_provider 제외
                class_name = module.__all__[0]
                provider_class = getattr(module, class_name)
                self.providers[file.stem] = provider_class()

    def get_provider(self, name):
        return self.providers.get(name)