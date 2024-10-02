# Function to load data
def load_data(train_path, test_path):
    try:
        sales_train_processed = pd.read_csv(train_path)
        sales_test_processed = pd.read_csv(test_path)
        return sales_train_processed, sales_test_processed
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


# Function to preprocess data
def preprocess_data(sales_train, sales_test):
    sales_train['date'] = pd.to_datetime(sales_train['date'])
    sales_test['date'] = pd.to_datetime(sales_test['date'])

    daily_sales_train = sales_train.groupby('date')['revenue'].sum().reset_index()
    daily_sales_train.rename(columns={'date': 'ds', 'revenue': 'y'}, inplace=True)

    daily_sales_test = sales_test.groupby('date')['revenue'].sum().reset_index()
    daily_sales_test.rename(columns={'date': 'ds', 'revenue': 'y'}, inplace=True)

    return daily_sales_train, daily_sales_test


# Function to create holidays DataFrame
def create_holidays_dataframe(sales_train):
    holidays = sales_train[['event_name', 'date']].dropna().drop_duplicates()
    holidays['ds'] = pd.to_datetime(holidays['date'])
    holidays['holiday'] = holidays['event_name']  # Use event names as holiday names
    holidays = holidays[['ds', 'holiday']]  # Keep only necessary columns
    return holidays


# Function to fit the Prophet model with holidays
def fit_prophet_model(train_data, holidays):
    model = Prophet(holidays=holidays)
    model.fit(train_data)
    return model


# Function to evaluate the model
def evaluate_model(test_data, forecast_data):
    mae = mean_absolute_error(test_data['y'], forecast_data['yhat'])
    rmse = mean_squared_error(test_data['y'], forecast_data['yhat'], squared=False)
    mape = mean_absolute_percentage_error(test_data['y'], forecast_data['yhat']) * 100

    return mae, rmse, mape

# Function to plot forecast
def plot_forecast(actual, forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(actual['ds'], actual['y'], label='Actual Sales', marker='o')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Sales', marker='x')
    plt.title('Actual vs Predicted Sales with Holidays')
    plt.xlabel('Date')
    plt.ylabel('Sales Revenue')
    plt.legend()
    plt.show()


# Plot the loss (MAE) over iterati
def plot_hyperopt_results(trials):
    results = pd.DataFrame({
        'iteration': range(len(trials.results)),
        'loss': [trial['result']['loss'] for trial in trials.trials]
    })
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results, x='iteration', y='loss', marker='o')
    plt.title('Hyperopt Optimization: Loss (MAE) Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.show()


# Plot the distributions of hyperparameters tried by Hyperopt
def plot_hyperparameter_distributions(trials):
    # Extract the hyperparameters from the trials
    params = pd.DataFrame([{
        'n_estimators': trial['misc']['vals'].get('n_estimators', [None])[0],
        'max_depth': trial['misc']['vals'].get('max_depth', [None])[0],
        'learning_rate': trial['misc']['vals'].get('learning_rate', [None])[0],
        'subsample': trial['misc']['vals'].get('subsample', [None])[0],
        'colsample_bytree': trial['misc']['vals'].get('colsample_bytree', [None])[0]
    } for trial in trials.trials])

    plt.figure(figsize=(15, 10))
    
    # Plot n_estimators
    plt.subplot(2, 3, 1)
    sns.histplot(params['n_estimators'], bins=10)
    plt.title('n_estimators Distribution')
    
    # Plot max_depth
    plt.subplot(2, 3, 2)
    sns.histplot(params['max_depth'], bins=10)
    plt.title('max_depth Distribution')
    
    # Plot learning_rate
    plt.subplot(2, 3, 3)
    sns.histplot(params['learning_rate'], bins=10)
    plt.title('learning_rate Distribution')
    
    # Plot subsample
    plt.subplot(2, 3, 4)
    sns.histplot(params['subsample'], bins=10)
    plt.title('subsample Distribution')
    
    # Plot colsample_bytree
    plt.subplot(2, 3, 5)
    sns.histplot(params['colsample_bytree'], bins=10)
    plt.title('colsample_bytree Distribution')
    
    plt.tight_layout()
    plt.show()