import requests
import zipfile
import os
import csv
from .haiku_validator import HaikuValidator


class CSVLoader:
    def __init__(self):
        self.validator = HaikuValidator()

    def load_default_dataset(self):
        url = "https://www.kaggle.com/api/v1/datasets/download/hjhalani30/haiku-dataset"
        download = requests.get(url)

        zip_path = "dataset/source/haiku-dataset.zip"
        extract_path = "dataset/haiku/all-haikus.csv"
        file_path = "dataset/haiku/valid-haikus.csv"

        with open("dataset/source/haiku-dataset.zip", "wb") as file:
            file.write(download.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            file_list = [f for f in zip_ref.namelist() if not f.endswith("/")]
            with zip_ref.open(file_list[0]) as source:
                content = source.read()
                with open(os.path.join(extract_path), "wb") as target:
                    target.write(content)

        with open(extract_path, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            next(csv_reader)
            valid_haikus = []
            # invalid_haikus = []
            for row in csv_reader:
                haiku = row[1] + "\n" + row[2] + "\n" + row[3]
                if self.validator.is_haiku(haiku):
                    valid_haiku = [row[1].strip(), row[2].strip(), row[3].strip()]
                    valid_haikus.append(valid_haiku)

        with open(file_path, "w", encoding="utf-8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            csv_writer.writerow(["first_line", "second_line", "third_line"])
            csv_writer.writerows(valid_haikus)
