# Importing the Spam messages data
import pandas as pd
data = pd.read_csv('SMSSpamCollection', sep = '\t', names = ['label','message'] )

# Data Cleaning and Pre-processing
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
Lemma = WordNetLemmatizer()
corpus = []

for i in range (len(data)):
    review = re.sub('[^a-zA-z]',' ', data['message'][i])
    review = review.lower()
    review = review.split()
    review = [Lemma.lemmatize(word)for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Generating the TF-IDF vector
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(max_features=5000)
X = vec.fit_transform(corpus).toarray()

#Converting the categorical target columns into integer values
y = pd.get_dummies(data['label'],drop_first=True)

#Model Training
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 0)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)

#Model Prediction
y_pred = clf.predict(X_test)

#Checking the model Accuracy
from sklearn.metrics import accuracy_score,confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)

# We can see that we achieve upto 97% accuracy with the Test dataset.
# So, Our spam classifier working very well.







