import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 names=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"])
print(df)

print(df.describe())

fig = px.scatter(df, x="SepalWidthCm", y="SepalLengthCm", color="Species", size='PetalLengthCm',
                 hover_data=['PetalWidthCm'])

fig2 = px.bar(df, x="SepalWidthCm", y="SepalLengthCm", color="Species")

fig3 = px.box(df, y="PetalWidthCm", color="Species", points='all')

fig4 = px.violin(df, y="PetalLengthCm", color="Species", violinmode='overlay', hover_data=df.columns)

fig5 = px.ecdf(df, x="SepalLengthCm", y="SepalWidthCm", color="Species", ecdfnorm=None)

X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(X_train)

scale = StandardScaler()
scale.fit(X_train)
X_train_sc = scale.transform(X_train)
X_test_sc = scale.transform(X_test)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train_sc, y_train)
predictor = clf.predict(X_test_sc)

pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier())])
pipeline.fit(X_train_sc, y_train)
r2 = pipeline.score(X_test_sc, y_test)
print(f"RFR: {r2}")

svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train_sc, y_train)

predictor_svc = svclassifier.predict(X_test_sc)

print(confusion_matrix(y_test, predictor_svc))
print(classification_report(y_test, predictor_svc))

logreg = LogisticRegression()
logreg.fit(X_train_sc, y_train)
predictor_reg = logreg.predict(X_test_sc)

print(confusion_matrix(y_test, predictor_reg))
print(classification_report(y_test, predictor_reg))
