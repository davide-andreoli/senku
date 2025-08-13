from rich.progress import (
    Progress,
    TimeRemainingColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
from typing import Generator, Tuple
from rich.table import Table
import pandas as pd

console = Console()


def display_training_progress(
    generator: Generator[Tuple[float, float, str], None, None],
):
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    overall_task = progress.add_task("Overall progress", total=100)
    epoch_task = progress.add_task("Epoch progress", total=100)

    status_text = "[green]Starting..."
    layout_group = Group(progress, Panel(status_text, border_style="green"))

    with Live(layout_group, refresh_per_second=10, console=console):
        for progress_overall, progress_epoch, status in generator:
            progress.update(overall_task, completed=progress_overall)
            progress.update(epoch_task, completed=progress_epoch)
            layout_group.renderables[1] = Panel(
                f"[green]{status}", border_style="green"
            )

        # Final update
        progress.update(overall_task, completed=100)
        progress.update(epoch_task, completed=100)
        layout_group.renderables[1] = Panel(
            "[bold green]Training complete!", border_style="green"
        )


def display_table_sample(sample: pd.DataFrame):
    table = Table(title="Sample")
    table.add_column("First line")
    table.add_column("Second line")
    table.add_column("Third line")

    for _, row in sample.iterrows():
        table.add_row(row["first_line"], row["second_line"], row["third_line"])
    console.print(table)
