# NEW CODE START
"""Phase 1: peek at the big parquet file without loading it all into memory."""

import os

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

load_dotenv(override=True)
PARQUET_PATH = os.getenv("CFPB_PARQUET_PATH")


def main() -> None:
    if not PARQUET_PATH:
        print("CFPB_PARQUET_PATH is missing from .env")
        return

    print("Reading:", PARQUET_PATH)
    parquet_file = pq.ParquetFile(PARQUET_PATH)

    # The file's "table of contents": tiny, no complaint data loaded
    print("Rows:", parquet_file.metadata.num_rows)
    print("Row groups (built-in chunks):", parquet_file.num_row_groups)
    print("Rows in first row group:", parquet_file.metadata.row_group(0).num_rows)
    print("Columns:", parquet_file.schema_arrow.names)

    # Load just ONE small column to see the date range
    dates = pd.read_parquet(PARQUET_PATH, columns=["date"])["date"]
    print("Earliest date:", dates.min())
    print("Latest date:", dates.max())


if __name__ == "__main__":
    main()
# NEW CODE END
