import pickle
import pandas as pd
# Load the model from the file
with open('my_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    
relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']

#User inputs
user_inputs = []
for feature in relevant_features:
    user_input = float(input(f"Enter the value for '{feature}': "))
    user_inputs.append(user_input)
    
#Creating df from user inputs
user_df = pd.DataFrame([user_inputs], columns = relevant_features)

#Predicting Anomalies using the model
user_pred = loaded_model.predict(user_df)

#Convert prediction values to binary values
user_pred_binary = 1 if user_pred == -1 else 0

if user_pred_binary == 1:
    print("Anomaly found: This transaction is marked as an anomaly.")
else:
    print("No Anomaly found: This transaction is normal.")