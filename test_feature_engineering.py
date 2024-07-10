from feature_engieering import preprocess_data, train_model, predict, add_feature
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def test_preprocess_data():
    # Sample input DataFrame
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    }
    df = pd.DataFrame(data)
    
    # Process the data
    df_processed = preprocess_data(df)
    
    # Print the processed DataFrame for debugging
    print("Processed DataFrame:")
    print(df_processed)
    
    # Check if the DataFrame is standardized
    mean_values = df_processed.mean().round(6)
    std_values = df_processed.std().round(6)
    
    # Print the mean and std values for debugging
    print("Mean values:")
    print(mean_values)
    print("Standard deviation values:")
    print(std_values)
    
    # Assert if the mean of each column is 0
    assert mean_values.eq(0).all(), f"Mean values are not 0: {mean_values}"
    
    # Assert if the standard deviation of each column is 1
    assert std_values.eq(1).all(), f"Standard deviation values are not 1: {std_values}"

# def test_preprocess_data():
#     # Sample input DataFrame
#     data = {
#         'feature1': [1, 2, 3, 4, 5],
#         'feature2': [10, 20, 30, 40, 50]
#     }
#     df = pd.DataFrame(data)
    
#     # Process the data
#     df_processed = preprocess_data(df)
    
#     # Check if the DataFrame is standardized
#     assert df_processed.mean().round(6).eq(0).all()
#     assert df_processed.std().round(6).eq(1).all()

def test_train_model():
    # Generate sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])
    
    # Train the model
    model, score = train_model(X, y)
    
    # Check if the model is a LogisticRegression instance
    from sklearn.linear_model import LogisticRegression
    assert isinstance(model, LogisticRegression)
    
    # Check if the score is a float
    assert isinstance(score, float)
    
    # Check if the score is within a valid range
    assert 0 <= score <= 1


def test_predict():
    # Generate sample data
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([0, 0, 1, 1, 1])
    X_test = np.array([[1, 2], [2, 3]])
    
    # Train a simple model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = predict(model, X_test)
    
    # Check if the predictions are of the correct type and shape
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    
    # Check if the predictions are within the expected range
    assert set(predictions).issubset({0, 1})

def test_add_feature():
    # Sample input DataFrame
    data = {
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    }
    df = pd.DataFrame(data)
    
    # Add new feature
    df_result = add_feature(df)
    
    # Expected output DataFrame
    expected_data = {
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'new_feature': [4, 10, 18]
    }
    expected_df = pd.DataFrame(expected_data)
    
    # Check if the result matches the expected output
    pd.testing.assert_frame_equal(df_result, expected_df)