import pandas as pd
from sklearn import metrics
from nltk.stem import WordNetLemmatizer
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.metrics import f1_score
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from langdetect import detect
from translate import Translator
import time
from sklearn.feature_selection import SelectKBest, f_classif


# **uncomment for the first run** downloads VADER sentiment analysis model
# nltk.download('vader_lexicon')

# Preprocess the text
def preprocess_text(text):

    # removes words beginning with '@' to remove Twitter handles
    text = re.sub(r'@\S+', ' ', text)
    
    # converts all text to lower case
    text = text.lower()

    # removes urls
    text = re.split('http.*', str(text))[0]
    
    # tokenize the text
    words = nltk.word_tokenize(text)
    
    # remove stop words
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # removes words with less than 4 characters
    words = [word for word in words if len(word) > 3]

    # perform lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

def map_labels(data):

    # map all 'fake' labels to 0 and all real labels to '1'
    label_mapping = {"fake": 0, "real": 1}

    # apply the mapping to the labels column
    data['label'] = data['label'].map(label_mapping)

    return data

def preprocess_train_data(data):
    # convert all 'humour' labels into 'fake'
    data.loc[(data.label == 'humor'),'label'] = 'fake'

    data = map_labels(data)

    # remove tweets with a length over 350
    data = data[data['tweetText'].str.len() <= 350]

    # apply the pre-processing function to every tweet
    data["tweetText"] = data["tweetText"].apply(preprocess_text)

    # remove pre-processed tweets that do not have a length of at least 3
    data = data[data['tweetText'].str.len() >= 3]

    # translate the text - not currently using this
    # data = translate_data(data)

    # remove duplicate rows
    data.drop_duplicates(subset=['tweetText'], keep='first', inplace=True, ignore_index = False)

    # gather the top 4 most occuring usernames
    top_users = data['username'].value_counts().sort_values(ascending=False).head(4).index
    # create a mask to remove tweets that are from these top 4 users
    mask = ~data['username'].isin(top_users)
    data = data[mask]

    return data

def translate_data(data):
    # currently unused code to translate the tweets to english if it is possible
    translator = Translator(to_lang="en")
    for index, row in data.iterrows():
        try:
            lang = detect(row["tweetText"])
            if lang != 'en':
                translation = translator.translate(row["tweetText"])
                data.at[index, 'tweetText'] = translation
        except:
            print("couldn't translate or detect: " + row["tweetText"])
    return data

def select_train_features(X_train, y_train, vectorizer, selector):

    # this will fit the vectorizer to the training data 
    # X_train is now a sparse matrix of TF-IDF features
    X_train = vectorizer.fit_transform(X_train).toarray()

    # creates a new dataframe, ensuring the columns of it match the feature names from the vectorizer for simple identification for later stages
    X_train = pd.DataFrame(X_train , columns=vectorizer.get_feature_names())

    # similar to the vectorizer, it fits the feature selector model using the training data
    X_train = selector.fit_transform(X_train, y_train)

    return X_train

def select_test_features(X_test, vectorizer, selector):
    # transforms the new data based on the fitted vectorizer from the previous fitting stage
    X_test = vectorizer.transform(data_test["tweetText"]).toarray()

    # creates a new dataframe, ensuring the columns of it match the feature names from the vectorizer for simple identification for later stages
    X_test = pd.DataFrame(X_test , columns= vectorizer.get_feature_names())

    # similar to the vectorizer, it transforms the new data based on the fitted selector from the previous fitting stage
    X_test = selector.transform(X_test)

    return X_test


# start the recording of process time used
start_time = time.process_time()

# read the dataset file and store it in dataframe
data = pd.read_csv("mediaeval-2015-trainingset.txt", sep="	")
# read the test dataset file and store it in dataframe
data_test = pd.read_csv("mediaeval-2015-testset.txt", sep="	")

# pre-process the training data
data = preprocess_train_data(data)

# pre-process the testing data - differently
data_test = map_labels(data_test)
data_test["tweetText"] = data_test["tweetText"].apply(preprocess_text)

# initialise the VADER sentiment analysis tool
sia = SentimentIntensityAnalyzer()
# create a new column in the data to store the sentiment of each tweet - whilst adding 1 to keep it within 0 and 2 instead of -1 and 1
data["sentiment_score"] = data["tweetText"].apply(lambda x: sia.polarity_scores(x)["compound"] + 1)
data_test["sentiment_score"] = data_test["tweetText"].apply(lambda x: sia.polarity_scores(x)["compound"] + 1)

# extract data needed for training and testing
X_train = data["tweetText"]
y_train = data['label']
X_test = data_test["tweetText"]
y_test = data_test['label']

# initialise the TF-IDF vectorizer with the specified parameters
vectorizer = TfidfVectorizer(max_features=40000, lowercase=False, analyzer='char', ngram_range=(3,5))

# initialise the select k best feature selection, with the number of features being 4000
# f_classif is one of the sklearn methods to determine revelance of a feature
selector = SelectKBest(f_classif, k=4000)

# extract the features for both the training and testing sets
X_train = select_train_features(X_train, y_train, vectorizer, selector)
X_test = select_test_features(X_test, vectorizer, selector)

# concatenate the features from the best k features and the sentiment analysis
X_train = np.hstack((X_train, data["sentiment_score"].values.reshape(-1,1)))
X_test = np.hstack((X_test, data_test["sentiment_score"].values.reshape(-1,1)))

# instantiate the Multinomial Naive Bayes classifier
clf = MultinomialNB()

# fit the classifier using the training data
clf.fit(X_train, y_train)

# predict the classification of the testing data
test_pred = clf.predict(X_test)

# stop the process timer as the core computation is complete
time_taken = time.process_time() - start_time

# prints the results of the testing data predicts against the testing data labels
print(metrics.classification_report(y_test, test_pred))
print("F1 score: ", f1_score(y_test, test_pred))
conf_matrix = confusion_matrix(y_test, test_pred)
tn, fp, fn, tp = conf_matrix.ravel()
print("TP: " + str(tp))
print("FP: " + str(fp))
print("TN: " + str(tn))
print("FN: " + str(fn))
print("Total elapsed: " + str(time_taken) + " seconds")
