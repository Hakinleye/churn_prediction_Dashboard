from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd


def calc_data():
    # Load customer data
    ### YOUR CODE HERE ### Step 1.3
    customer_data = pd.read_csv('customer_data.csv')

    # Display the first few rows of the dataset
    print(customer_data.head())


   # Fill missing values and encode categorical columns
    ### YOUR CODE HERE ### Step 1.4
    customer_data['tenure'] = customer_data['tenure'].fillna(customer_data['tenure'].mean())


    state_encoder = LabelEncoder()
    ### YOUR CODE HERE ### Step 1.5
    # Encode 'state'
    customer_data['encoded_state'] = state_encoder.fit_transform(customer_data['state'])

 # Store the mapping from encoded state to original state names
    state_mapping_inverse = {label: original_state for label, original_state in enumerate(state_encoder.classes_)}

    # Code for contract_renewal encoding 
    contract_renewal_encoder = LabelEncoder()
    customer_data['encoded_contract_renewal'] = contract_renewal_encoder.fit_transform(customer_data['contract_renewal'])
    
 
   
     # Scale numerical columns
    scaler = StandardScaler()
    ### YOUR CODE HERE ### Step 1.6
    customer_data[['tenure_scaled', 'monthly_charges_scaled']] = scaler.fit_transform(customer_data[['tenure', 'monthly_charges']])
    print(customer_data.head())
   
    

    # Split the data into features and target for model training
    # Use the scaled features and the NEWLY encoded 'contract_renewal'
    x_full = customer_data[['tenure_scaled', 'monthly_charges_scaled', 'encoded_state', 'encoded_contract_renewal']] # Added 'encoded_contract_renewal'
    y_full = customer_data['churn']

    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=42)

   
    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    

    ### YOUR CODE HERE ### Step 2.1
    # Fit the model to the training data
    model.fit(x_train, y_train)

    ### YOUR CODE HERE ### Step 2.2
    # Make predictions on the test data
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)[:, 1]

    # Evaluate the model
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

   
  
    # Average churn probability (on test set)
    ### YOUR CODE HERE ### Step 3.1.
    # Predict probabilities for x_test
    #customer_data['predicted_churn_prob'] = model.predict_proba(x_full)[:, 1]
    #avg_churn_prob = customer_data['predicted_churn_prob'].mean()
    test_probs = model.predict_proba(x_test)[:, 1]
    avg_churn_prob = test_probs.mean()
  
   
    
     # Calculate average churn probability (from test set probabilities)
            ### YOUR CODE HERE ### Step 3.2
    # High-risk customers: predicted to churn (y_pred == 1) AND above avg prob
    high_risk_customers = customer_data['predicted_churn_binary'].sum() #> avg_churn_prob).sum()

   

    # Predict churn probability for entire dataset
    customer_data['predicted_churn_prob'] = model.predict_proba(x_full)[:, 1]
    # Flag high-risk customers using avg churn from test set
    customer_data['predicted_churn_binary'] = (customer_data['predicted_churn_prob'] > avg_churn_prob).astype(int)

    customer_data['actual_churn'] = y_full # Actual churn for calculating churn rate per state
   
    # Create lists to store churn rates and high-risk counts by state
    churn_rate_by_state = []
    high_risk_by_state = []

     # Group by 'encoded_state' using the full customer_data DataFrame
    for encoded_state_val, group in customer_data.groupby('encoded_state'):
        original_state_name = state_mapping_inverse.get(encoded_state_val, "Unknown State")

        # Pass actual churn (for churn rate) and binary predictions (for high-risk count)
        churn_rate, high_risk_count = calculate_churn_and_high_risk(
            group['actual_churn'],          # For Step 3.3 (churn_rate)
            group['predicted_churn_binary'] # Now using binary prediction for high-risk count
        )
        # Append the state and churn rate to the list
        churn_rate_by_state.append({'state': original_state_name, 'churn_rate': churn_rate})
        # Append the state and high risk count to the list
        high_risk_by_state.append({'state': original_state_name, 'high_risk': high_risk_count})

    # Convert lists to DataFrames
    churn_rate_by_state_df = pd.DataFrame(churn_rate_by_state)
    high_risk_by_state_df = pd.DataFrame(high_risk_by_state)
   

    print("Results of your analysis for reference:")
    print(f"Average Churn Probability: {avg_churn_prob}")
    print(f"High-Risk Customers: {high_risk_customers}")
    print(f"Churn Rate by State:\n {churn_rate_by_state_df}")
    print(f"High-Risk Customers by State:\n {high_risk_by_state_df}")

    # Store the results in a text file for autograding. Do not modify this code.
    with open('churn_results.txt', 'w') as f:
        f.write("Do not modify this file. It is used for autograding the processed data from the lab.\n\n")
        f.write(f"Average Churn Probability: {avg_churn_prob}\n\n")
        f.write(f"High-Risk Customers: {high_risk_customers}\n\n")
        f.write(f"Churn Rate by State:\n {churn_rate_by_state_df}\n\n")
        f.write(f"High-Risk Customers by State:\n {high_risk_by_state_df}")
    
    return avg_churn_prob, high_risk_customers, churn_rate_by_state_df, high_risk_by_state_df

def calculate_churn_and_high_risk(actual_churn_series, predicted_binary_series):
    """Calculate churn rate and count high-risk customers for a given state group."""
    ### YOUR CODE HERE ### Step 3.3
   
       ### YOUR CODE HERE ### Step 3.4
    
    

if __name__ == "__main__":
    calc_data()