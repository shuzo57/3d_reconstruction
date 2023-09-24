import numpy as np
import pandas as pd
from tabulate import tabulate  # type: ignore


def calculate_delay_frame(df1, df2, name) -> int:
    df1.interpolate(method="linear", limit_direction="both", inplace=True)
    df2.interpolate(method="linear", limit_direction="both", inplace=True)

    df1_y = df1[f"{name}_y"] - df1[f"{name}_y"].mean()
    df2_y = df2[f"{name}_y"] - df2[f"{name}_y"].mean()

    corr = np.correlate(df1_y, df2_y, "full")
    delay_frame = int(corr.argmax() - (len(df2_y) - 1))

    return delay_frame


def delay_frame_viz(csv_file1, csv_file2, name) -> None:
    df1 = pd.read_csv(csv_file1, index_col="frame")
    df2 = pd.read_csv(csv_file2, index_col="frame")

    delay_frame = calculate_delay_frame(df1, df2, name)

    if delay_frame > 0:
        table = [[csv_file1, 1 + delay_frame], [csv_file2, 1]]
    else:
        table = [[csv_file1, 1], [csv_file2, 1 - delay_frame]]

    print(tabulate(table, headers=["Video", "Frame"], tablefmt="grid"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate delay between two videos"
    )
    parser.add_argument(
        "-f1", "--csv_file1", type=str, help="Path to the first CSV file"
    )
    parser.add_argument(
        "-f2", "--csv_file2", type=str, help="Path to the second CSV file"
    )
    parser.add_argument("-n", "--name", type=str, help="Name of the variable")
    args = parser.parse_args()

    delay_frame_viz(args.csv_file1, args.csv_file2, args.name)
