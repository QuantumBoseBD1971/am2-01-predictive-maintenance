"""Download the public UCI dataset used by the project."""

from predictive_maintenance.data import download_dataset

if __name__ == "__main__":
    path = download_dataset()
    print(f"Dataset written to: {path}")
