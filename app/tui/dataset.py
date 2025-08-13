from app.core.dataset import load_default_dataset
from rich import print
from app.core.rich import display_table_sample


def load_dataset_flow():
    print("\nLoading default dataset...\n")
    stats, sample, _ = load_default_dataset()
    print(f"{stats}")

    display_table_sample(sample)
