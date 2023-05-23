#importing libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
import pickle

#Loading the data
data=pd.read_csv("IRIS.csv")

#Separating function parameters and target parameters
X=data[['sepal_length','sepal_width','petal_length','petal_width']]
Y=data['species']

#Split dataset into train and test
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=5,test_size=0.3)

#Model creation
model=LogisticRegression()

#fitting the model to data
model.fit(x_train,y_train)

prediction=model.predict(x_test)



# making pickle file of model
pickle.dump(model,open('model.pkl','wb'))



