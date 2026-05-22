import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Load dataset
df = pd.read_csv(
    "spam.csv",
    sep="\t",
    names=["label","message"]
)

print("\nDataset Loaded Successfully\n")

print(df.head())


# Features and target
X = df["message"]
y = df["label"]


# Convert text to numbers
cv = CountVectorizer()

X = cv.fit_transform(X)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model = MultinomialNB()

model.fit(X_train,y_train)

print("\nModel Training Complete")


# Prediction
prediction = model.predict(X_test)


# Evaluation
accuracy = accuracy_score(
    y_test,
    prediction
)

print("\nAccuracy:")

print(accuracy)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        prediction
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        prediction
    )
)


# Test custom message

while True:

    user_message=input(
        "\nEnter a message (type exit to stop): "
    )

    if user_message.lower()=="exit":
        break

    message=cv.transform(
        [user_message]
    )

    result=model.predict(
        message
    )

    print(
        "\nPrediction:",
        result[0]
    )