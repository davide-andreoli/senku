from core.dataset import load_default_dataset
from rich import print
from rich.table import Table


def load_dataset_flow():
    print("\nLoading default dataset...\n")
    stats, sample, _ = load_default_dataset()
    print(f"{stats}")

    table = Table(title="Sample")
    table.add_column("First line")
    table.add_column("Second line")
    table.add_column("Third line")

    for _, row in sample.iterrows():
        table.add_row(row["first_line"], row["second_line"], row["third_line"])

    print(table)
