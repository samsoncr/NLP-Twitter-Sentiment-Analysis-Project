![VCU Logo](https://ocpe.vcu.edu/media/ocpe/images/logos/bm_CollEng_CompSci_RF2_hz_4c.png)

# NLP-Twitter-Sentiment-Analysis-Project
| Developer Name | VCU Email Address | Github Username |
| :---: | :---: | :---: |
| Charles Cutler | cutlerci@vcu.edu | cutlerci |
| Christopher Smith | samsoncr@vcu.edu | samsoncr |

# WHEN WE ARE READY TO MAKE A TABLE OF CONTENTS
http://ecotrust-canada.github.io/markdown-toc/

## READ ME SHOULD CONTAIN THE FOLLOWING

* Project description

The objective of this project is to perform sentiment analysis on tweets from Twitter. Sentiment analysis is a classification problem in which our code will be able to predict the sentiment of a tweet given the text of the tweet. Sentiment is an attribute of a tweet that ranges from highly positive to highly negative; however, for the purpose of this project, sentiment is divided into two classes: positive or negative. Therefore, given any tweet, our code will label it as either positive or negative. Then we must analyze the accuracy of our model. We do this by comparing the model’s predictions to labels made by humans which are assumed to be correct.
The model for predicting sentiment is a convolutional neural network (CNN). This is a type of machine learning algorithm. It requires two steps. First, we must preprocess a large training dataset. The data used for this project can be found in the data folder with more description in the README.md in the data folder.
The dataset contains tweets and information about the tweets including their man-made labels. The tweets must be preprocessed before being vectorized and embedded. After its transformation, the data can then be given to the CNN which will learn from data how to predict sentiment.
We then collect the predictions of our model on a test set of data, and for our results we calculate the precision, recall, F1, and accuracy scores of our predictions. Further detail is found in the sections below.


* Installation instructions
  * Explicitly show how to install your code in point-by-point fashion
  * I will copy and paste the instructions onto the command line for installation
  * If you are using a jupyter notebook, please do not state “open jupyter notebook” – explain to a novice how to do this 
* Usage instructions
  * How do I run your code – Keep in mind I am used to running my programs on the command line and I don’t use an IDE – so how do I run your code? 

* Method 
  * What method did you use?
  * What feature representation(s) did you explore?
  * What algorithm(s) did you use?
  * A picture is worth a 1000 words
* Data
  * What is your data? - DONE
  * How many labels do you have? - DONE
  * What is your train/dev/test split? - DONE
  * How many instances in your train/dev/test split? - DONE
  * You can graph it. - DONE
* Results
  * What are the precision, recall, F1 and accuracy scores of your models? 
  * You can put those in a graph or table
* Discussion 
  * Analysis of your results
  * What worked?
  * What didn’t work?
  * Why? 
* Future Work 
  * What would you do next given all the time and resources in the world? 


# Method
## Preprocessing:
Doing some research in the general practices for sentiment analysis we chose to do the following preprocessing steps:
* Removal of special characters, punctuation, and digits
* Removal of usernames and URLs 
* Repeated Character Reduction 
* Contraction Expansion
* Stop word removal
* Lemmatization
* Capitalization removal

These allowing the data from within a tweet to be more uniform and also reduce the amount of "noise" or not helpful data that gets input into the model.

## Feature Extraction and Vectorization
Once the data was cleaned it is necessary to turn the words of the tweet into a numerical representations that can be used to train a machine learning model. The steps to build these numerical representations are as follows:
* Build some tools
* Convert the tweets
* Feed the input to the CNN

### Build some tools: Static Word Embeddings with Glove
To determine the numerical representations of tweets it is neccesary to first define som euseful tools. We used ``GloVe: Global Vectors for Word Representation`` to construct two tools that together can be used to convert each word found within a tweet into a numerical vector. The collection of all the individual word vectors then represent the numerical representation of a tweet. 

The first tool is the ``embedding array``. This array consists of the individual word vectors that are in the GloVe database. A snippet of this array is shown below:

![EmbeddingsArray](./EmbeddingsArray.png "Embeddings Array")
 
 The second tool that goes right along with the first one is the ``index mapping``. This tool allows for the word in the tweets to be converted into a single dimensional numerical vector that acts and the intermediary between words and word embeddings. It contains the index location for every word that is in the Embeddings Array. For example, the word representation for ``Dogs`` might be stored in the second line in the ``Embeddings Array``. So in the ``Index Mapping`` the word ``Dogs`` would be matched with the number 2, representing its storage location in ``Embeddings Array``. A snipped of this mapping is shown below:
 
![IndexMapping](./IndexMapping.png "Index Mapping")
 
To get a better idea of this process of converting a tweet of words into a numerical representation let's look at an example:

### Feature extraction and Vectorization Example	
    Here is an example tweet:
    “I like fluffy dogs”
    
    Given we have built the Embeddings Array and stored the locations of every word in ``Index Mappings``:
    We could assume that the indexes of the static word embeddings, that is those numerical word representations, are:
    
    I : 29 
    like: 99
    fluffy: 746
    dogs: 2
    
    So we can now build an index mapped versions of out tweet:
    [ 29 99 746 2 ]
    
    Later on, when we pass Embeddings Array and our index version of the tweet into the embedding layer of the Convolutional Neural Network,
    [ 29 99 746 2 ] turns into
    [ [ 3.15530002e-01  5.37649989e-01  1.01769999e-01  3.25529985e-02 … 3.79800005e-03 ]
    [ 1.53639996e-02 -2.03439996e-01  3.32940012e-01 -2.08859995e-01  … 1.00610003e-01 ]
    [ 3.09760004e-01  5.00150025e-01 3.20179999e-01  1.35370001e-01  … 8.70389957e-03 ]
    [ 1.91100001e-01 2.46680006e-01 -6.07520007e-02 -4.36230004e-01  … 1.93019994e-02 ] ]

## Sentiment detection / Classification using machine learning
Convolutional Neural network
We used the following layers:

![Layers](./Layers.png "CNN Layers")


# Data 
## Original Data
We used the Sentiment140 dataset that can be found online at: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download)
This dataset contains tweets annotated for sentiment analysis. They use two labels, specifically a 0 label for negative tweets and a 4 label for positive tweets. There are the exact same number of positive tweets as negative tweets making this a balanced dataset. This dataset consists of six pieces of information stored for each collected tweet. In order from left to right the pieces are: 
1) Target: The polarity of the tweet ( 0 = negative, 4 = positive ) 
2) Ids: The id of the tweet ( ex: 2087 )
3) Date: The date of the tweet ( ex: Sat May 16 23:58:44 UTC 2009 )
4) Flag: The query (ex: lyx). If there is no query, then this value is NO_QUERY.
5) User: The user that tweeted ( ex: robotickilldozr )
6) Text: The text of the tweet ( ex: Lyx is cool )
This data can be found in the Data sub-directory as "sentiment140Dataset.zip"

## Cleaned Data
After using the "preproccessor.py" python script that we wrote we obtained a cleaner version of the Sentiment140 dataset. Specifically we did the following preproccesing tasks to clean the data into the format we wanted for training our model:
1) Extract only columns 1, 2, and 6 from the original dataset and discard the rest of the columns. That is we only keep the ID, annotated sentiment, and text of the tweet.
2) Remove URLs from a tweet
3) Remove Mentions from a tweet
4) Contraction Expansion
5) Remove Numbers from a tweet
6) Remove Symbols from a tweet
7) Reduce any repeated more than twice chracters into just three ``ex: Loooooooooveeeeee becomes loooveee``
8) Remove StopWords from a tweet 
9) Lemmatize the words in a Tweet

This clean version of the dataset can be found in the Data sub-directory as "Cleaned_Sentiment140_Data.csv"

We split the data set into three pieces for use in training and evaluation of the model. We use an 80% train, 10% development or validation, and 10% test split. Numerically that break down looks as follows:

![DataSplitGraph](./DataSplit.png "Data Split Graph")

# Results
The following table displays results we obtained when training the model with the entire training data subset.
![Results](./OverallResults.png "Overall Results")

We were interested in how much data we actually need to train a model to do well. To investigate this we conducted an ablation study, using smaller and smaller subsets of the training data to train a model. The following sections describe the results we found. 
## Confusion Matrix
![ConfusionMatrix](./ConfusionMatrix.png "Confusion Matrix")

## Training Versus Validation Loss
![TrainingVsValidationLoss](./TrainingVsValidationLoss.png "Training Vs Validation Loss")

## Precision, Recall, and F1 Performance Measures
![PrecisionRecallandF1](./P_R_F1.png "Precision, Recall, and F1 Performance Measures")
