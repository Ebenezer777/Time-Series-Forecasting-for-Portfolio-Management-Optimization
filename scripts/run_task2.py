from src.modeling import load_tsla, train_test_split, run_arima, evaluate

def main():

    print("Loading Tesla data...")
    series = load_tsla()

    train, test = train_test_split(series)

    print("Training ARIMA...")
    forecast, order = run_arima(train, test)

    mae, rmse, mape = evaluate(test, forecast)

    print("\nARIMA Order:", order)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE:", mape)

if __name__ == "__main__":
    main()
