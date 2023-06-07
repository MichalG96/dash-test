import datetime
from enum import StrEnum, auto

import pandas as pd
from pathlib import Path

import typer

app = typer.Typer()


class TimeOfDay(StrEnum):
    morning = auto()
    evening = auto()


@app.command()
def migrate(
    start_date: datetime.datetime = typer.Option(...),
    tod: TimeOfDay = typer.Option("morning"),
):
    start_date = start_date.date()

    rename_files("data/data.parquet", "data/data_backup.parquet")

    old_df = pd.read_parquet("data/data_backup.parquet")
    new_df = pd.read_csv("data_current.csv")
    merged_df = pd.concat([old_df, new_df])
    merged_df.to_parquet("data/data.parquet")

    create_skeleton(start_date, tod)


@app.callback()
def callback():
    pass


def rename_files(old_name: str, new_name: str) -> None:
    path = Path(old_name)
    path.replace(new_name)


def create_skeleton(start_date: datetime.date, tod: TimeOfDay) -> None:
    skeleton = pd.DataFrame(
        {
            "date": pd.date_range(
                start=start_date,
                end=start_date + datetime.timedelta(days=30),
                freq="12H",
            )[:-1]
        }
    )
    skeleton["date"] = pd.to_datetime(skeleton["date"]).dt.date

    skeleton["time_of_day"] = ["morning", "evening"] * (len(skeleton) // 2)
    skeleton["weight"] = None
    if tod == "evening":
        skeleton = skeleton.iloc[1:, :]
    skeleton.to_csv("data_current.csv", index=False)


if __name__ == "__main__":
    app()
