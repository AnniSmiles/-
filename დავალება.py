import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#1
file_path = 'C:\Users\akats\OneDrive\Desktop\csv files\pizza_sales.csv'
pizza_data = pd.read_csv(file_path)

# # select  columns for regression model
# X = pizza_data[['quantity']]
# y = pizza_data['total_price']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# model = LinearRegression()

# # train model 
# model.fit(X_train, y_train)


# y_pred = model.predict(X_test)

# mae = metrics.mean_absolute_error(y_test, y_pred)
# mse = metrics.mean_squared_error(y_test, y_pred)
# rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
# r2 = metrics.r2_score(y_test, y_pred)

# print(f'mean absolute Error: {mae}')
# print(f'mean squared Error: {mse}')
# print(f'root mean squared error: {rmse}')
# print(f'r-squared: {r2}')

# #predict the total price for a new order with a quantity of 5
# new_data = pd.DataFrame({'quantity': [5]})
# predicted_total_price = model.predict(new_data)

# print(f'predicted price: {predicted_total_price[0]}')

# #2
# X = pizza_data[['quantity', 'pizza_size']]

# column_transformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
# X = column_transformer.fit_transform(X)

# y = pizza_data['total_price']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = LinearRegression()


# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# mae = metrics.mean_absolute_error(y_test, y_pred)
# mse = metrics.mean_squared_error(y_test, y_pred)
# rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
# r2 = metrics.r2_score(y_test, y_pred)

# # print(f'mean absolute Error: {mae}')
# # print(f'mean squared Error: {mse}')
# # print(f'root mean squared error: {rmse}')
# # print(f'r-squared: {r2}')


# # predict total price for a new order with quantity 5 and pizza size 'medium'
# new_data = pd.DataFrame({'quantity': [5], 'pizza_size': ['medium']})

# new_data_encoded = column_transformer.transform(new_data)

# predicted_total_price = model.predict(new_data_encoded)

# print(f'predicted  price  {predicted_total_price[0]}')

#3

# X = pizza_data[['quantity', 'unit_price']]
# y = pizza_data['total_price']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# model = DecisionTreeRegressor(random_state=42)


# model.fit(X_train, y_train)


# y_pred = model.predict(X_test)


# mae = metrics.mean_absolute_error(y_test, y_pred)
# mse = metrics.mean_squared_error(y_test, y_pred)
# rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
# r2 = metrics.r2_score(y_test, y_pred)


# # print(f'mean absolute Error: {mae}')
# # print(f'mean squared Error: {mse}')
# # print(f'root mean squared error: {rmse}')
# # print(f'r-squared: {r2}')


# #predict the total price for a new order with quantity 5 and unit price 8.5
# new_data = pd.DataFrame({'quantity': [5], 'unit_price': [8.5]})
# predicted_total_price = model.predict(new_data)

# print(f'predicted  price  {predicted_total_price[0]}')

# #4
# #we are trying to figure out if new order will be vegetarian or not
# X = pizza_data[['quantity', 'unit_price']] 
# y = (pizza_data['pizza_category'] == 'vegetarian').astype(int)  # convert to binary (1 for vegetarian, 0 for non-vegetarian)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = LogisticRegression(random_state=42)

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)


# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)

# print(f'accuracy: {accuracy}')
# print(f'confusion matrix:\n{conf_matrix}')
# print(f'classification report:\n{classification_rep}')

# #predict the category for a new order with quantity 5 and unit price 8.5
# new_data = pd.DataFrame({'quantity': [5], 'unit_price': [8.5]})
# predicted_category = model.predict(new_data)

# print(f'Predicted Category: {"vegetarian" if predicted_category[0] == 1 else "non-vegetarian"}')

# #5
# #predit category of pizza
# X = pizza_data[['quantity', 'unit_price']]
# y = pizza_data['pizza_category'] 

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# model = DecisionTreeClassifier(random_state=42)


# model.fit(X_train, y_train)


# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)


# print(f'accuracy: {accuracy}')
# print(f'confusion matrix:\n{conf_matrix}')
# print(f'classification report:\n{classification_rep}')

# #predict the category for a new order with quantity 5 and unit price 8.5
# new_data = pd.DataFrame({'quantity': [5], 'unit_price': [8.5]})
# predicted_category = model.predict(new_data)

# print(f'Predicted Category: {predicted_category[0]}')

