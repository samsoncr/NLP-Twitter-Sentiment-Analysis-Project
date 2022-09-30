# We used the Sentiment140 dataset that can be found online at https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download
# The data in each row of this data set is, in order from left to right: 
    # target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
    # ids: The id of the tweet ( 2087)
    # date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
    # flag: The query (lyx). If there is no query, then this value is NO_QUERY.
    # user: the user that tweeted (robotickilldozr)
    # text: the text of the tweet (Lyx is cool)

# �


import csv
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def contradictionExpansion(uncleanText):
    # Contraction Expansion RegEx -- Found here:
    # https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
    uncleanText = re.sub(r"won\'t", "will not", uncleanText)
    uncleanText = re.sub(r"can\'t", "can not", uncleanText)
    uncleanText = re.sub(r"n\'t", " not", uncleanText)
    uncleanText = re.sub(r"\'re", " are", uncleanText)
    uncleanText = re.sub(r"\'s", " is", uncleanText)
    uncleanText = re.sub(r"\'d", " would", uncleanText)
    uncleanText = re.sub(r"\'ll", " will", uncleanText)
    uncleanText = re.sub(r"\'t", " not", uncleanText)
    uncleanText = re.sub(r"\'ve", " have", uncleanText)
    uncleanText = re.sub(r"\'m", " am", uncleanText)
    return uncleanText



def symbolRemoval(uncleanText):
    uncleanText = uncleanText.replace(')', "")
    uncleanText = uncleanText.replace('(', "")
    uncleanText = uncleanText.replace(':', "")
    uncleanText = uncleanText.replace(';', "")
    uncleanText = uncleanText.replace('%', "")
    uncleanText = uncleanText.replace('!', "")
    uncleanText = uncleanText.replace('*', "")
    uncleanText = uncleanText.replace('£', "")
    uncleanText = uncleanText.replace('¬', "")
    uncleanText = uncleanText.replace('°', "")
    uncleanText = uncleanText.replace('|', "")
    uncleanText = uncleanText.replace('|', "")
    uncleanText = uncleanText.replace('@', "")
    uncleanText = uncleanText.replace('¥', "")
    uncleanText = uncleanText.replace('&', "")
    uncleanText = uncleanText.replace('é', "")
    uncleanText = uncleanText.replace('´', "")
    uncleanText = uncleanText.replace('', "")
    uncleanText = uncleanText.replace('{', "")
    uncleanText = uncleanText.replace('}', "")
    uncleanText = uncleanText.replace('â', "")
    uncleanText = uncleanText.replace('á', "")
    uncleanText = uncleanText.replace('¶', "")
    uncleanText = uncleanText.replace('¯', "")
    uncleanText = uncleanText.replace('®', "")
    uncleanText = uncleanText.replace('è', "")
    uncleanText = uncleanText.replace('[', "")
    uncleanText = uncleanText.replace(']', "")
    uncleanText = uncleanText.replace('+', "")
    uncleanText = uncleanText.replace('=', "")
    uncleanText = uncleanText.replace('§', "")
    uncleanText = uncleanText.replace('ù', "")
    uncleanText = uncleanText.replace('ã', "")
    uncleanText = uncleanText.replace('²', "")
    uncleanText = uncleanText.replace('¾', "")
    uncleanText = uncleanText.replace('¹', "")
    uncleanText = uncleanText.replace('½', "")
    uncleanText = uncleanText.replace('ð', "")
    uncleanText = uncleanText.replace('µ', "")
    uncleanText = uncleanText.replace('ñ', "")

    return uncleanText

    
# Set of stop words from NLTK
stops = set(stopwords.words('english'))
# Extra stop words I wanted to include.
extraStops = {'', 'The', 'In', '<s>', '</s>', '<@>', '</p>', '<p>', '...',
              '.', ",\"", ')', '(', ".\"", ".\")", ',', "\"", ':', ';', '?', '%','..','!','*', '£', '¬', '°', '|', '|','@', '¥', '&', 'é', '´', '', '{'}
# Complete Set of Stop Words
stop_words = stops.union(extraStops)

# Open the comma seperated file containing the data from the Sentiment140 tweet dataset
# WHY DOES LATIN-1 FIX utf error ?!?!?!
with open('training.1600000.processed.noemoticon.csv') as inputDataFile:
    # Use the built in CSV reader to read from the file containing the tweet data
    csvreader = csv.reader(inputDataFile)

    rows = []
    
    for rowOfData in csvreader:
        # Retrieve useful information from each data row
        classifiedSentiment = rowOfData[0]
        tweetId = rowOfData[1] 
        uncleanText = rowOfData[-1]

        # Remove URLs
        uncleanText = re.sub(r"(http|https|ftp)://[a-zA-Z0-9\\./]+.", "", uncleanText)

        # Remove Mentions
        uncleanText = re.sub(r"@(\w+).","",uncleanText)

        # Contraction Expansion
        uncleanText = contradictionExpansion(uncleanText)

        # Remove Numbers
        uncleanText = re.sub(r"[0-9]+", "", uncleanText)

        # Remove Symbols
        uncleanText = symbolRemoval(uncleanText)

        # Reduce any repeated more than twice chracters into just three
        # Example Loooooooooveeeeee becomes loooveee
        uncleanText = re.sub(r"(.[^\.])\1{2,}","\1\1\1", uncleanText)
        
        # Tokenize the tweet
        uncleanTokens = word_tokenize(uncleanText)

        # Remove StopWords
        filtered_tokens = [w for w in uncleanTokens if not w.lower() in stop_words]

        # Lemmatize the words in a Tweet
        lemmatizer = WordNetLemmatizer()
        cleanTextTokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

        # Construct a cleaned version of a tweet's text
        cleanText = ""
        for token in cleanTextTokens:
            cleanText += token.lower() + " "

        # Remove the trailing space
        cleanText = cleanText.rstrip(" ")  

        rows.append([classifiedSentiment,tweetId,cleanText])


# Output the cleaned Data to a .csv file using pandas
cleanedData = rows
header = ["Sentiment", "Tweet Id", "Cleaned Tweet Text"]
cleanedDataFrame = pd.DataFrame(cleanedData, columns=header)
cleanedDataFrame.to_csv('Cleaned_Sentiment140_Data.csv', index=False)
