import csv
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import gensim.downloader as api

# Using the following model for static word embeddings
# https://huggingface.co/Gensim/glove-twitter-25
# Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation
model = api.load("glove-twitter-200")
print("Model has loaded.")

assignedSentiment = {}
rows = []

with open('Cleaned_Sentiment140_Data.csv') as inputDataFile:
    count = 0
    # Use the built in CSV reader to read from the file containing the tweet data
    csvreader = csv.reader(inputDataFile)
    # Skip over the header of the CSV
    next(csvreader)

    for rowOfData in csvreader:
        count += 1

        # Initialize an empty tweet vector
        tweetVector = np.zeros(200)
    
        # Retrieve useful information from each data row
        classifiedSentiment = rowOfData[0]
        tweetId = rowOfData[1] 
        tweetText = rowOfData[2]

        # Add the sentiment to a Dictionary with the key as the tweetId
        assignedSentiment[str(tweetId)] = classifiedSentiment

        # Tokenize the tweet
        tweetTokens = word_tokenize(tweetText)
        numberOfTokens = len(tweetTokens)

        # Get unigrams and static word embeddings from the glove model
        # Create the sum of all static word embeddings of words found 
        for token in tweetTokens:
            try:
                modelVector = model[token]
            except KeyError:
                modelVector = np.zeros(200)
            tweetVector = np.column_stack((tweetVector, modelVector)) 
        try:
            tweetVector = np.mean(tweetVector, axis=1)
        except np.AxisError:
            tweetVector = tweetVector + np.zeros(200)

        if count % 10 == 0:
            print(count)
        if count == 16000001:
            break
        rows.append([classifiedSentiment,tweetId,list(tweetVector)])

# Output the cleaned Data to a .csv file using pandas
cleanedData = rows
header = ["Sentiment", "Tweet Id", "Vector Representation of Tweet"]
cleanedDataFrame = pd.DataFrame(cleanedData, columns=header)
cleanedDataFrame.to_csv('Sentiment140_Vector_Representation.csv', index=False)
