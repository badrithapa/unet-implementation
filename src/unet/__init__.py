from .model import UNet

__all__ = ["UNet"]


def main() -> None:
    print(f"Hello from {UNet.__name__} module!")
