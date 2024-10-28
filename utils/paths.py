from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


DATA = HERE / "data" / "datasets"

CONFIG = HERE / "configs"


# NOTE: debug
print(f"DEDUG - #Project Path HERE: {HERE}")
