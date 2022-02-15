import pandas as pd
#Reading excel dataset
df = pd.read_excel('ENB2012_data.xlsx')
newcolumn_names = {'X1':'Relative_Compactness', 'X2': 'Surface_Area', 
                'X3':  'Wall_Area', 'X4': 'Roof_Area', 'X5': 'Overall_Height',
                'X6': 'Orientation', 'X7': 'Glazing_Area', 
                'X8': 'Glazing_Area_Distribution', 
                'Y1': 'Heating_Load', 'Y2': 'Cooling_Load'}

#Rename column headers from X and Y's to Feature names
df = df.rename(columns=newcolumn_names)

#loading the dataset as a DataFrame
dataframe = pd.DataFrame(df, columns=df.columns)

#Loading the dataset as a normalized DataFrame with columns as columns of dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(dataframe), columns = df.columns)

#To get features dataset/dataframe alone i.e and remove the Heating load and Cooling load columns (output columns)
#Use .drop()
features_df = normalized_df.drop(columns=['Heating_Load', 'Cooling_Load'])
print(features_df)

#To get dataframe of just output value 'Heating Load'
heating_target = normalized_df['Heating_Load']

#Splitting the data into training and testing set using train_test_split
from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(features_df, heating_target, test_size = 0.3, random_state = 1)

#Calling the linear Regression model
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()

#Fitting the model to the training data
linear_model.fit(x_train, y_train)

#To obtain predictions
predicted_values = linear_model.predict(x_test)
print(predicted_values)


#To determine the Mean Absolute Error (MAE)
    #remember that MAE is sum of absolute difference between Predicted output and actual output
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predicted_values)
print(mae) # answer is 0.054374251510641756
print (round(mae, 3)) #answer is 0.054 to 3dp


#To determine the Residual sum of Squares (RSS) of the Model
import numpy as np
from sklearn.metrics import mean_squared_error
#rmse stands for residual mean squared error
rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
print(rmse) #answer is 0.07493000442813244
print(round(rmse, 3)) #answer is 0.075


#To determine the R-Squared of the model
from sklearn.metrics import r2_score
R_Squared = r2_score(y_test, predicted_values)
print(R_Squared) #answer is 0.9235738125443352
print(round(R_Squared, 3)) #answer is 0.924


#To determine the Ridge Regression
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.5)
ridge_reg.fit(x_train, y_train)


#To determine Lasso Regression
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(x_train, y_train)


#Comparing the effects of both regularization
    #First obtain the weight of every feature as a dataframe per model using the function below
def get_weights_df(model, features_df, column_name):
#Recall that weights can be obtained as a list using model.coef_
#feats.column will return a list of column names of dataframe feats
    weights= pd.Series(model.coef_, index=features_df.columns)
#To sort the weight values in the series in ascending order
    weights= pd.Series(model.coef_, index=features_df.columns).sort_values()
#reset_index will make the new index 0,1,2,3 and make a column named index of the former index
    weights_df=pd.DataFrame(weights).reset_index()
    weights_df.columns=['Features', column_name]
#Returns dataframe having 0,1,2,3 index of model weight per feature   
    return weights_df


#To get model weight per feature for all three models; Linear Regression model, Ridge Regression model and Lasso Regression model
linear_model_weights_df = get_weights_df(linear_model, x_train, 'Linear_Model_Weight')
ridge_weights_df = get_weights_df(ridge_reg, x_train, 'Ridge_Weight')
lasso_weights_df = get_weights_df(lasso_reg, x_train, 'Lasso_Weight')
print(linear_model_weights_df)

#To merge all three dataframes into one
    #First we merge the first two into one using the column features which they have in common and the on parameter
final_weights= pd.merge(linear_model_weights_df, ridge_weights_df, on='Features')
    #Next we merge the newly merged and the last one into one using the same column features
final_weights= pd.merge(final_weights, lasso_weights_df, on='Features')
print(final_weights)
