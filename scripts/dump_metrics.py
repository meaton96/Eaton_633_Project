from shared_util.fetch import fetch_all_metrics
from pathlib import Path

if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "src" / "shared_util" / "data"
    df = fetch_all_metrics()
    print(f' fetched rows: {len(df)}')
    df.to_csv(data_path / "metrics.csv", index=False)