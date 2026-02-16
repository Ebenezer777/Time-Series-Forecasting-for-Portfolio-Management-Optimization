from src.data_loader import download_data, save_processed

def main():
    print("Downloading financial data...")
    data = download_data()
    save_processed(data)
    print("Data saved in data/processed/")

if __name__ == "__main__":
    main()
