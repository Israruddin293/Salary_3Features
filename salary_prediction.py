import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Assuming the file is in the same directory as your script
data = pd.read_csv('dataset.csv')

data.columns = [col.strip() for col in data.columns]
print(data.columns)

# Extract features (YearsExperience, Education, ExperienceRating) and target variable (Salary)
X = data[['YearsExperience', 'Education', 'ExperienceRating']]
y = data['Salary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the results (modify the plot according to your needs)
plt.scatter(X_test['YearsExperience'], y_test, color='black', label='Actual Data')
plt.scatter(X_test['YearsExperience'], y_pred, color='blue', marker='x', label='Predicted Data')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Salary Prediction')
plt.legend()
plt.show()

# Now you can use the trained model to predict salary for new data
user_experience = float(input("Enter years of experience: "))
user_education = float(input("Enter education level: "))
user_experience_rating = float(input("Enter experience rating: "))

new_data_point = [[user_experience, user_education, user_experience_rating]]

# Use the trained model to predict salary for the user's input
predicted_salary = model.predict(new_data_point)
print(f'Predicted Salary: {predicted_salary[0]}')


# Plot the new data point and its predicted salary on the graph
plt.scatter(user_experience, predicted_salary, color='red', marker='x', s=100, label='Predicted Salary')
plt.legend()

# Show the plot
plt.show()

