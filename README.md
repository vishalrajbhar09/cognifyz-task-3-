# cognifyz-task-3-

Restaurant Cuisine Classification Using Machine Learning
This project is a machine learning-based approach to classify restaurants into different cuisine types based on features like average cost for two, price range, and votes. The program uses the Random Forest Classifier for classification. 


Objective: Predict the type of cuisine (e.g., Italian, Chinese, Indian) served by a restaurant based on specific features.
Dataset: The dataset contains details of restaurants, including Cuisines, Average Cost for two, Price range, and Votes.
Model Used: A Random Forest Classifier is used to build the prediction model.
Evaluation: The performance of the model is evaluated using accuracy and a classification report.

Steps in the Code

Step 1: Import Necessary Libraries
The code imports the following Python libraries:

pandas: For data manipulation and analysis.
sklearn modules:
train_test_split: For splitting the dataset into training and testing sets.
RandomForestClassifier: For building the classification model.
LabelEncoder: For encoding the target variable (cuisine names) into numerical values.
classification_report, accuracy_score: For evaluating model performance.

Step 2: Load the Dataset
The dataset is loaded using pd.read_csv() by specifying the file path.
Input: A CSV file containing restaurant information.
Example columns in the dataset:
Cuisines: Cuisine types (e.g., Italian, Indian, Chinese).
Average Cost for two: The average cost for two people.
Price range: A numeric representation of price levels (e.g., 1, 2, 3).
Votes: Number of votes for the restaurant.

Step 3: Handle Missing Values
Missing values in the Cuisines column are replaced with 'Unknown' using fillna() to ensure the dataset is clean.

Step 4: Encode the Target Variable
Target Variable: The column Cuisines is the target variable, which is encoded into numeric labels using LabelEncoder.

Example:
Indian → 10
Italian → 11
Chinese → 12
A new column Cuisine_Label is created to store the encoded values.

Step 5: Select Features (X) and Target (y)
Features (X): Selected columns that influence the cuisine type:
Average Cost for two
Price range
Votes
Target (y): The Cuisine_Label column (numerical representation of cuisines).

Step 6: Split the Dataset
The data is split into training and testing sets using an 80-20 ratio.
Training data: 80% (for training the model).
Testing data: 20% (for evaluating the model).

Step 7: Train the Classification Model
A Random Forest Classifier is used with 100 decision trees (n_estimators=100).
The model is trained on the training data (X_train, y_train) using model.fit()

Step 8: Evaluate the Model
Predictions are made on the test set (X_test) using model.predict().
Accuracy:

The model's performance is evaluated using accuracy_score(), which gives the percentage of correct predictions.

 exampe (Accuracy: 0.85) 
Classification Report:
The report includes metrics like precision, recall, F1-score, and support for each cuisine class.
Example Output:
markdown
Copy code
Classification Report:
         
Step 9: Analyze Feature Importance
The feature importance of each input feature (e.g., Votes, Price range) is analyzed using model.feature_importances_.
This helps identify which features are most influential for predicting the cuisine type.

