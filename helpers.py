import pickle
from sklearn.metrics import r2_score

with open('modelplus.pkl', 'rb') as file:
    loaded_structures = pickle.load(file)

# Retrieve the necessary structures from the loaded dictionary
loaded_model = loaded_structures['model']
loaded_X_train = loaded_structures['X_train']
loaded_y_train = loaded_structures['y_train']
loaded_X_test = loaded_structures['X_test']
loaded_y_test = loaded_structures['y_test']

# Make predictions on the test data using the loaded model
y_pred = loaded_model.predict(loaded_X_test)

# Calculate the R2 score
r2 = r2_score(loaded_y_test, y_pred)

print(f"R2 score: {r2}")
