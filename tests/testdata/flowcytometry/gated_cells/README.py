# To create the 3 csv files cellmat.csv, ybinary.csv and yternary.csv I run this code
import os
import re

import pandas as pd

gated_cells_dir = os.path.join(
    "data",
    "shared",
    "rhskl1",
    "01_raw",
    "01_FlowCytometryData",
    "UKR_Hutchinson",
    "2020-12-17_GatedCells",
)
# 1. Loading the data:
# 1.1 Cell subtype classification
loaded_dict_gated = {}
for class_x in os.listdir(gated_cells_dir):
    # print(class_x)
    for file_x in os.listdir(os.path.join(gated_cells_dir, class_x)):
        csv_type = re.sub(r"([^_]*)_.*", r"\1", file_x)
        with open(os.path.join(gated_cells_dir, class_x, file_x)) as f:
            for i, l in enumerate(f):
                pass
        n = i + 1  # number of records in file
        s = 1000  # desired sample size
        import random

        skip = sorted(random.sample(range(1, n), n - s))
        print(skip)
        loaded_csv = pd.read_csv(
            os.path.join(gated_cells_dir, class_x, file_x), index_col=0, skiprows=skip
        )
        print(loaded_csv.columns)
        loaded_csv.to_csv(csv_type + ".csv", index=False, float_format="%.2f")
    exit()
