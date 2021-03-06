from importlib import import_module

import pandas as pd
import streamlit as st
from sklearn import datasets

st.write("""# Simple Iris Flower Prediction app""")

st.sidebar.header("User Input Parameters")

# classifiers and their dotted module path
classifiers = {
    "RandomForestClassifier": "sklearn.ensemble",
    "GradientBoostingClassifier": "sklearn.ensemble",
    "KNeighborsClassifier": "sklearn.neighbors",
    "RadiusNeighborsClassifier": "sklearn.neighbors"
}


def import_classifier(classifier):
    """
    This function takes classifier name and
     returns its imported class
    :param classifier:
    :return:
    """
    return getattr(
        import_module(classifiers[classifier]),
        classifier
    )


def get_input_features():
    sepal_length = st.sidebar.slider("Sepal length", 0.0, 10.0, 5.4)
    sepal_width = st.sidebar.slider("Sepal width", 0.0, 10.0, 3.4)
    petal_length = st.sidebar.slider("Petal length", 0.0, 10.0, 1.3)
    petal_width = st.sidebar.slider("Petal width", 0.0, 10.0, 0.2)
    classifier = st.sidebar.selectbox("Select Classifier", list(classifiers.keys()), 0)

    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    return pd.DataFrame(data, index=[0]), classifier


# Example DF
#    sepal_length  sepal_width  petal_length  petal_width
# 0           5.4          3.4           1.3          0.2
df, selected_classifier = get_input_features()

st.subheader("User Input parameters")
st.write(df)

iris = datasets.load_iris()
x = iris.data
y = iris.target

imported_classifier_class = import_classifier(selected_classifier)
clf = imported_classifier_class()
clf.fit(x, y)

prediction = clf.predict(df)
prediction_probability = clf.predict_proba(df)

st.subheader("Class labels and their corresponding index number")
st.write(iris.target_names)

# Predicted species 
st.subheader("Predicted species")
st.write(iris.target_names[prediction])

# Prediction Probability
st.subheader("Prediction Probability")
st.write(prediction_probability)
