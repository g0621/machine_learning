import pandas as pd
import matplotlib.pyplot as plt


def confusiontest(cm):
    # https://towardsdatascience.com/precision-vs-recall-386cf9f89488
    TN, FP, FN, TP, = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_Score = 2 * Precision * Recall / (Precision + Recall)
    print("Accuracy : " + str(Accuracy) + "\nPrecision : " + str(Precision))
    print("Recall : " + str(Recall) + "\nF1 Score : " + str(F1_Score))


# we use tsv because natural language mostly contains the comma

df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# cleaning the text

# steaming -> changing all the words with same meaning to same root
# like loving,loved,loves to love

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# nltk.download('stopwords')
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()

    # removing stop words and steaming

    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df['Liked'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# from sklearn.preprocessing import StandardScaler
#
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# building the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
