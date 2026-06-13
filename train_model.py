import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics import accuracy_score

df = pd.read_csv("StudentsPerformance.csv")

df["average_score"] = (
    df["math score"] +
    df["reading score"] +
    df["writing score"]
) / 3

df["success"] = (df["average_score"] >= 60).astype(int)

le_gender = LabelEncoder()
le_lunch = LabelEncoder()
le_test = LabelEncoder()

df["gender"] = le_gender.fit_transform(df["gender"])
df["lunch"] = le_lunch.fit_transform(df["lunch"])
df["test preparation course"] = le_test.fit_transform(
    df["test preparation course"]
)

X = df[
    [
        "gender",
        "lunch",
        "test preparation course",
        "math score",
        "reading score",
        "writing score"
    ]
]

y = df["success"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

pickle.dump(model, open("model.pkl", "wb"))

print(f"Model Accuracy: {accuracy*100:.2f}%")
print("Model Saved Successfully!")