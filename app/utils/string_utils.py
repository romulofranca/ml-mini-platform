import os
import re


def normalize_dataset_name(file_name: str) -> str:
    """
    Normalizes a dataset file name by:
      - Removing the file extension.
      - Converting the name to lower case.
      - Replacing any non-alphanumeric characters (including spaces) with
        underscores.
      - Removing any leading or trailing underscores.
    """
    base = os.path.splitext(file_name)[0]
    base = base.lower()
    base = re.sub(r"\W+", "_", base)
    base = base.strip("_")
    return base
