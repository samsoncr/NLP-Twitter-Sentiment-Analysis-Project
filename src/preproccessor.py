# Developers: Charles Cutler and Christopher Samson
# Class: CMSC 516 Advanced Natural Language Processing
# The following code was developed for the first programming assignment in the course
# CMSC 516, Advanced Natural Language Processing, at Virginia Commonwealth University
#
#
# We used the Sentiment140 dataset that can be found online at 
#   https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download
#
# The data in each row of this data set is, in order from left to right: 
    # target: the polarity of the tweet (0 = negative, 4 = positive)
    # ids: The id of the tweet ( 2087)
    # date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
    # flag: The query (lyx). If there is no query, then this value is NO_QUERY.
    # user: the user that tweeted (robotickilldozr)
    # text: the text of the tweet (Lyx is cool)

# If you do not use Google Colab, make sure to install these python libraries.
# Installation instructions can be found at:
#
# https://pandas.pydata.org/docs/getting_started/install.html
# https://www.nltk.org/install.html

import csv
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# A custom built function to simply data preprocessing for each tweet
# This function expands contractions such as "can't" into "can not"
def contractionExpansion(uncleanText):
    # Contraction Expansion RegEx -- The source for these Regular Expressions can be here:
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


# A custom built function to simply data preprocessing for each tweet
# This function removes a large collection of unwanted symbols from the text of each tweet
# Some of these were chosen from knowledge a priori some of these were chosen after manually
# going through sections of the database
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

    
# Set of stop words from NLTK. Often these words, called stop words, are not helpful when trying to "learn" something from text 
# and thus removed from each tweet
stops = set(stopwords.words('english'))
# Extra stop words I wanted to include.
extraStops = {'', 'The', 'In', '<s>', '</s>', '<@>', '</p>', '<p>', '...',
              '.', ",\"", ')', '(', ".\"", ".\")", ',', "\"", ':', ';', '?', '%','..','!','*', '£', '¬', '°', '|', '|','@', '¥', '&', 'é', '´', '', '{'}
# Complete Set of Stop Words
stop_words = stops.union(extraStops)

# Open the comma seperated file containing the data from the Sentiment140 tweet dataset
with open('training.1600000.processed.noemoticon.csv') as inputDataFile:

    # Use the built in CSV reader to read from the file containing the tweet data
    csvreader = csv.reader(inputDataFile)

    rows = []
    
    for rowOfData in csvreader:

        # Retrieve useful information from each data row, specifically:
        classifiedSentiment = rowOfData[0] # The annotated sentiment
        tweetId = rowOfData[1]             # The unique tweet id
        uncleanText = rowOfData[-1]        # The unprocessed text

        # Many of the following preprocessing tasks are accomplished using regular expressions.
        # To learn more about regular expressions, go here: https://en.wikipedia.org/wiki/Regular_expression

        # Remove URLs from a tweet
        uncleanText = re.sub(r"(http|https|ftp)://[a-zA-Z0-9\\./]+.", "", uncleanText)

        # Remove Mentions from a tweet
        uncleanText = re.sub(r"@(\w+).","",uncleanText)

        # Contraction Expansion using custom function
        uncleanText = contractionExpansion(uncleanText)

        # Remove Numbers from a tweet
        uncleanText = re.sub(r"[0-9]+", "", uncleanText)

        # Remove Symbols from a tweet
        uncleanText = symbolRemoval(uncleanText)

        # Reduce any repeated more than twice chracters into just three
        # Example Loooooooooveeeeee becomes loooveee
        uncleanText = re.sub(r"(.[^\.])\1{2,}","\1\1\1", uncleanText)
        
        # Tokenize a tweet
        uncleanTokens = word_tokenize(uncleanText)

        # Remove StopWords from a tweet
        filtered_tokens = [w for w in uncleanTokens if not w.lower() in stop_words]

        # Lemmatize the words in a Tweet
        # This means that we reduce words into their root form and helps 
        # reduce the number of unique words in tweets
        lemmatizer = WordNetLemmatizer()
        cleanTextTokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

        # Reconstruct a cleaned version of a tweet's text
        cleanText = ""
        for token in cleanTextTokens:
            cleanText += token.lower() + " "

        # Remove the trailing space that is found after reconstruction
        cleanText = cleanText.rstrip(" ")  

        # Store the annotated sentiment, unqiue tweet ID, and the cleaned tweet text
        rows.append([classifiedSentiment,tweetId,cleanText])


# Output the cleaned Data to a .csv file using pandas
cleanedData = rows
header = ["Sentiment", "Tweet Id", "Cleaned Tweet Text"]
cleanedDataFrame = pd.DataFrame(cleanedData, columns=header)
cleanedDataFrame.to_csv('Cleaned_Sentiment140_Data.csv', index=False)
