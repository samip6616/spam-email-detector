#documentation in https://app.notion.com/p/Rajat-s-Page-3b41d3238156803b8aebcbace5b49efb?source=copy_link


import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Loading the dataset
df = pd.read_csv("dataset\spam_ham_dataset.csv",encoding='latin-1')

#Renaming the unnamaed column in dataset to index
df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
# print(df.columns) #shows the current coloumns of dataset

#  Changing the name from ham to not spam and spam to spam
#THIS WHOLE FUICKING SECTION IS NOT NEEDED CAUSE I AM RETARDED
# df['label']= df['label'].replace({
#     'ham' : 'not spam',
#     'spam': 'spam'
# }
# )  

# Input and target
X = df['text']       # Email text → INPUT
Y = df['label_num']  # 0/1 → TARGET

#train_test_split
(X_train,X_test,Y_train,Y_test)= train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# Convert text to numerical values
cv = CountVectorizer()
X_train_Vectorized = cv.fit_transform(X_train)

# Create and train model
model = MultinomialNB()
model.fit(X_train_Vectorized, Y_train)

# Test model
# X_train_vectorized = cv.fit_transform(X_train) //yo vayo vane kina chaldaina code idk why
X_train_vectorized = cv.transform(X_train)
X_test_vectorized= cv.transform(X_test)

print("\nTraining accuracy:", model.score(X_train_vectorized, Y_train))
print("\nTesting accuracy:", model.score(X_test_vectorized, Y_test))

# Predict new email (data)
new_email = [  #Yesma jeni hale hunxa test garna spam or not spam
    "Hello! My name is Rajat.",
    "Hi, how are you?",
    "Can we meet tomorrow?",
    "Congratulations! You have won a $1000 prize!",
    "Claim your free reward now!",
    "You have won a lottery. Click here to claim your prize."
]
new_email_vectorized = cv.transform(new_email)

prediction = model.predict(new_email_vectorized)

# print(prediction) //Yesle result 1 or 0 ko form ma dinxa

#output customization

# This part doenst fucking work if u have an array/list of elements
# if prediction[0] == 1:
#     print("spam")
# else:
#     print("not spam")

for email, prediction in zip(new_email, prediction):
    if prediction == 1:
        result = "\nspam"
    else:
        result = "\nnot spam"

    print(f"{result}: {email}")