# TASK 3 - IRIS FLOWER CLASSIFICATION
# CodSoft Data Science Internship
# Author: Aarthi S

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# STEP 1: LOAD IRIS DATASET (OFFICIAL DATASET)
# --------------------------------------------------
iris_data = load_iris()

# Convert dataset into DataFrame for clarity
features = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
labels = pd.Series(iris_data.target, name="Species")

# --------------------------------------------------
# STEP 2: SPLIT DATA INTO TRAINING & TESTING
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.25,
    random_state=42,
    shuffle=True
)

# --------------------------------------------------
# STEP 3: CREATE MACHINE LEARNING MODEL
# --------------------------------------------------
model = KNeighborsClassifier(
    n_neighbors=5,
    metric='minkowski'
)

# --------------------------------------------------
# STEP 4: TRAIN THE MODEL
# --------------------------------------------------
model.fit(X_train, y_train)

# --------------------------------------------------
# STEP 5: PREDICT USING TEST DATA
# --------------------------------------------------
predictions = model.predict(X_test)

# --------------------------------------------------
# STEP 6: EVALUATE MODEL PERFORMANCE
# --------------------------------------------------
accuracy = accuracy_score(y_test, predictions)

print("IRIS FLOWER CLASSIFICATION RESULT")
print("---------------------------------")
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=iris_data.target_names
))