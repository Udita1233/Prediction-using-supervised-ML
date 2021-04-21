# Prediction-using-supervised-ML

# importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb
%matplotlib inline\

# loading the required dataset
dataset= pd.read_csv("C:/Users/pc/Desktop/DATA.csv")
print("Data imported successfully")

#viewing the data
dataset.head(10)

#removing the spaces
dataset.columns= dataset.columns.str.strip()

#gives information about dataset
dataset.info()

#generates descriptive statistics
dataset.describe()

#visualizing the data
#plotting the distribution of scores
sb.set_style('darkgrid')
sb.scatterplot(y= dataset['Scores'], x= dataset['Hours'])
plt.title('Hours vs Scores', size = 20)
plt.xlabel('Hours studied', size= 15)
plt.ylabel('Percentage scores', size=15)
plt.show()

#plotting the regression line and printing the correlation 
sb.regplot(x= dataset['Hours'], y= dataset['Scores'])
plt.title('Regression Line', size = 20)
plt.xlabel('Hours studied', size=15)
plt.ylabel('Percentage scores', size= 15)
plt.show()
print(dataset.corr())

#dividing the data into attributes(inputs) and labels(outputs)
X= dataset.iloc[:, :-1].values
Y= dataset.iloc[:, 1]. values

#splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)

print("X train.shape =", X_train.shape)
print("Y train.shape =", Y_train.shape)
print("X test.shape =", X_test.shape)
print("Y test.shape =", Y_test.shape)

#training the linear regression model
from sklearn.linear_model import LinearRegression    
regressor = LinearRegression()    
regressor.fit(X_train, Y_train) 
print("Training complete.")
print("Model trained.")

#predicting the scores for the model
pred_Y= regressor.predict(X_test)
prediction = pd.DataFrame({'Hours': [i[0] for i in X_test], 'Predicted Marks': [k for k in pred_Y]})
prediction

#comparing the actual versus predicted model
df= pd.DataFrame({'Actual': Y_test, 'Predicted': pred_Y})
df

#visualizing the actual vs predicted by plotting
plt.scatter(x= X_test, y= Y_test, color='Red')
plt.plot(X_test, pred_Y, color= 'Black')
plt.title('Actual vs Predicted', size=20)
plt.xlabel('Hours studied', size=12)
plt.ylabel('Marks Percentage', size=12)
plt.show()

#predicting the score for 9.25 hours of study
hours= [9.25]
pred= regressor.predict([hours])
print("The score of the student when he studies for 9.25 hours a day = {}".format(round(pred[0],3)))

#evaluating the model and getting the error
from sklearn import metrics
print('Mean Absolute error: ',
     metrics.mean_absolute_error(Y_test, pred_Y))
