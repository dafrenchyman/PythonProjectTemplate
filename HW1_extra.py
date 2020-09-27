
# THIS HAS THE PROFESSORS COMMENTS
# I had to switch my directories, in order for my data to be pulled from the right folder
# Find out your current working directory 
import os
print(os.getcwd())
#The below line changed the file path to my homework folder 
os.chdir(r"C:\Users\Trini\Documents\SDSU Big Data Analytics\Machine Learning\HW 1\Download files")

# import pandas, numpy and plotly
import pandas as pd
import numpy as np
import plotly.express as px

# read your data into VS Code from 'filename .csv'
data_iris = pd.read_csv("iris_data.csv")

# Preview the first 5 lines of the loaded data 
data_iris.head

# descriptive stats
data_iris.describe()

##################################################################################################
#                                          Scatter-Plot                                          #
##################################################################################################

data_iris.columns = ["sepal_length", "Sepal_Width", "Petal_Length", "Petal_Width", "Class"]
fig = px.scatter(data_iris, x="Sepal_Width", y="sepal_length", color="Class")
fig.show()

##################################################################################################
#                                          whisker box-plot                                      #
##################################################################################################
data_iris.columns = ["sepal_length", "Sepal_Width", "Petal_Length", "Petal_Width", "Class"]
data_iris.head
# add residual error column 
data_iris["e"] = data_iris["Sepal_Width"]/100
# check how it looks
data_iris.head
fig = px.scatter(data_iris, x="Sepal_Width", y="sepal_length", color="Class", error_x="e", error_y="e")
fig.show()


##################################################################################################
#                                          bar graph                                             #
##################################################################################################

# read in original data --> without the "e"
data_iris = pd.read_csv("iris_data.csv")
data_iris.head
data_iris.columns = ["Sepal_Length", "sepal_width", "petal_length", "petal_width", "class"]
fig = px.bar(data_iris, x="sepal_width", y="Sepal_Length", color="class", barmode= "group")
fig.show()


##################################################################################################
#                                          Scatter Plot Matrices                                 #
##################################################################################################

# check my data 
data_iris.head
data_iris.columns = ["Sepal_Length", "sepal_width", "petal_length", "petal_width", "Iris class"]
fig = px.scatter_matrix(data_iris, dimensions=["sepal_width", "Sepal_Length"
                                    , "petal_width", "petal_length"], color="Iris class")
fig.show()


##################################################################################################
#                                          Density Contour                                       #
##################################################################################################

data_iris.columns = ["Sepal_Length", "sepal_width", "petal_length", "petal_width", "species"]
fig = px.density_contour(data_iris, x="sepal_width", y="Sepal_Length")
fig.show()

##################################################################################################
#                                      Scikit Learn a go-go                                      #
##################################################################################################

from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
# Normalizer doesn't work with strings so kept it to numerical data
X = data_iris [["Sepal_Length", "sepal_width", "petal_length", "petal_width"]]
transformer = Normalizer().fit(X)
#set norm to mean of sepal length and transform it by that
#transformer = Normalizer(norm=X["Sepal_Length"].mean).transform(X)
transformer
X_trans = transformer.transform(X)
X_trans
#tried to adjust this to # of rows and columns
#X_trans, y = make_classification(n_samples=149, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
X_trans, y = make_classification(n_samples=149, n_features=4)
clf = RandomForestClassifier(max_depth=4, random_state=0)
clf.fit(X_trans,y)
print(clf.predict([[1,0,0,0]]))


##################################################################################################
#                                      Pipelining                                                #
##################################################################################################

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
X_trans, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_trans, y, random_state=0)
pipe = Pipeline([('scaler', Normalizer()), ('svc', SVC())])

# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
### I split the data to avoid data leackage. Not that it matters but it runs :p 

pipe.fit(X_train, y_train)

pipe.score(X_test, y_test)


No commit comments for this range
© 2020 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
