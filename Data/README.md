![VCU Logo](https://ocpe.vcu.edu/media/ocpe/images/logos/bm_CollEng_CompSci_RF2_hz_4c.png)

# NLP-Twitter-Sentiment-Analysis-Project
| Developer Name | VCU Email Address | Github Username |
| :---: | :---: | :---: |
| Charles Cutler | cutlerci@vcu.edu | cutlerci |
| Christopher Smith | samsoncr@vcu.edu | samsoncr |

# Data 
## Original Data
We used the Sentiment140 dataset that can be found online at: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download)
This data set consists of six pieces of information stored for each collected tweet. In order from left to right the pieces are: 
1) Target: The polarity of the tweet ( 0 = negative, 4 = positive ) 
2) Ids: The id of the tweet ( ex: 2087 )
3) Date: The date of the tweet ( ex: Sat May 16 23:58:44 UTC 2009 )
4) Flag: The query (ex: lyx). If there is no query, then this value is NO_QUERY.
5) User: The user that tweeted ( ex: robotickilldozr )
6) Text: The text of the tweet ( ex: Lyx is cool )
This data can be found in this directory as "sentiment140Dataset.zip"

## Cleaned Data
After using the "preproccessor.py" python script that we wrote we obtained a cleaner version of the Sentiment140 dataset. Specifically we did the following preproccesing tasks to clean the data into the format we wanted for training our model:
1) Extract only columns 1, 2, and 6 from the original dataset and discard the rest of the columns
2) Remove URLs from a tweet
3) Remove Mentions from a tweet
4) Contraction Expansion
5) Remove Numbers from a tweet
6) Remove Symbols from a tweet
7) Reduce any repeated more than twice chracters into just three ``ex: Loooooooooveeeeee becomes loooveee``
8) Remove StopWords from a tweet 
9) Lemmatize the words in a Tweet

This clean version of the dataset can be found in this directory as "Cleaned_Sentiment140_Data.csv"
