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
    

    ### YOUR CODE HERE ### Step 2.2
    # Make predictions on the test data
   

    # Evaluate the model
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

   
  
    # Average churn probability (on test set)
    ### YOUR CODE HERE ### Step 3.1.
   
  
   
    
     # Calculate average churn probability (from test set probabilities)
            ### YOUR CODE HERE ### Step 3.2
    # High-risk customers: predicted to churn (y_pred == 1) AND above avg prob
   

    # Predict churn probability for entire dataset
    
    # Flag high-risk customers using avg churn from test set
   
    # Create lists to store churn rates and high-risk counts by state
   

     # Group by 'encoded_state' using the full customer_data DataFrame
   
        # Pass actual churn (for churn rate) and binary predictions (for high-risk count)
       
                  # For Step 3.3 (churn_rate)
           
        # Append the state and churn rate to the list
        
        # Append the state and high risk count to the list
        

    # Convert lists to DataFrames
   

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