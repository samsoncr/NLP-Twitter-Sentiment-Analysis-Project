![VCU Logo](https://ocpe.vcu.edu/media/ocpe/images/logos/bm_CollEng_CompSci_RF2_hz_4c.png)

# NLP-Twitter-Sentiment-Analysis-Project
| Developer Name | VCU Email Address | Github Username |
| :---: | :---: | :---: |
| Charles Cutler | cutlerci@vcu.edu | cutlerci |
| Christopher Smith | samsoncr@vcu.edu | samsoncr |

# For creating the table of contents
http://ecotrust-canada.github.io/markdown-toc/


# Table of Contents
- [NLP-Twitter-Sentiment-Analysis-Project](#nlp-twitter-sentiment-analysis-project)
- [Project description](#project-description)
- [Installation instructions](#installation-instructions)
- [Usage instructions](#usage-instructions)
- [Method](#method)
  * [Preprocessing:](#preprocessing-)
  * [Feature Extraction and Vectorization](#feature-extraction-and-vectorization)
    + [Build some tools: Static Word Embeddings with Glove](#build-some-tools--static-word-embeddings-with-glove)
    + [Feature extraction and Vectorization Example](#feature-extraction-and-vectorization-example)
  * [Sentiment detection / Classification using machine learning](#sentiment-detection---classification-using-machine-learning)
- [Data](#data)
  * [Original Data](#original-data)
  * [Cleaned Data](#cleaned-data)
- [Results](#results)
  * [Confusion Matrix](#confusion-matrix)
  * [Training Versus Validation Loss](#training-versus-validation-loss)
  * [Precision, Recall, and F1 Performance Measures](#precision--recall--and-f1-performance-measures)
- [Discussion](#discussion)
- [Future Work](#future-work)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


# Project description

The objective of this project is to perform sentiment analysis on tweets from Twitter. Sentiment analysis is a classification problem in which our code will be able to predict the sentiment of a tweet given the text of the tweet. Sentiment is an attribute of a tweet that ranges from highly positive to highly negative; however, for the purpose of this project, sentiment is divided into two classes: positive or negative. Therefore, given any tweet, our code will label it as either positive or negative. Then we must analyze the accuracy of our model. We do this by comparing the model’s predictions to labels made by humans which are assumed to be correct. The model for predicting sentiment is a convolutional neural network (CNN), a type of machine learning algorithm.

The whole process begins by preprocessing a large training dataset. The data used for this project can be found in the Data sub directory. The dataset contains tweets and information about the tweets including their man-made labels. These tweets must be preprocessed before being vectorized and embedded. After its transformation, the data can then be given to the CNN which will learn from data how to predict sentiment. We then collect the predictions of our model on a test set of data, and for our results we calculate the precision, recall, F1, and accuracy scores of our predictions.

Further detail is found in the sections below.


# Installation instructions

Begin by downloading the VectorizationAndCNN.ipynb file. You can follow this tutorial for downloading a single file from github: https://www.wikihow.com/Download-a-File-from-GitHub

Follow the same instructions for downloading the cleaned data set file called Cleaned_Sentiment140_Data.csv.

We will also need the glove.twitter.27B.200d.txt file which can be downloaded from https://www.kaggle.com/datasets/fullmetal26/glovetwitter27b100dtxt. This file may be too big to download, however, so you can use this public google drive link https://drive.google.com/file/d/1R84p-LC9zun-pHflhexbrX743IlsM4P8/view?usp=sharing to upload the file to your drive. Click the link and use ZIP Extractor to extract the file to Google Drive.


<p align="center">
 <img src="./ZIP Extractor.PNG" width="1000" height="200">
</p>

You can skip the downloads of these two files if you already have downloaded the whole project as a zip file. This can be done by clicking on the green button that says "code" in the main directory of the github page (same directory as this README), and then clicking on "Download ZIP".

We recommend using Google Colab to run the code. To do so, go to google drive at drive.google.com. From your downloads on your local machine, upload the two files, VectorizationAndCNN.ipynb and Cleaned_Sentiment140_Data.csv, to your google drive. Do the same for glove.twitter.27B.200d.txt either from your own downloads or the previous link. Make sure they are uploaded to your MyDrive folder in Google Drive.


Incase you choose to run the code locally, you must install Python and the necessary packages for tensorflow, numpy, keras, and sklearn.
Tutorials for installing Python and these libraries can be found at these links:

https://realpython.com/installing-python/

https://www.tensorflow.org/install 

https://numpy.org/install/ 

https://www.tutorialspoint.com/keras/keras_installation.htm 

https://scikit-learn.org/stable/install.html 

# Usage instructions

Open the VectorizationAndCNN.ipynb file in your google drive. There are two lines of code that need to be uncommented before running in Google Colab. These are lines 26 and 27. You can also read through the comments for more guidance through this process.

Then click on the runtime tab, and click run all.

If you chose to run the file locally, you will need to instead replace the file path names on lines 78 and 79 with the local file paths. This also means that you will have to download the glove.twitter.27B.200d.txt file from https://www.kaggle.com/datasets/fullmetal26/glovetwitter27b100dtxt. Read through the comments at the top of the page in VectorizationAndCNN.ipynb for more details on changing the path names. You can export VectorizationAndCNN.ipynb as a .py file and then run it on the command line by navigating to the directory containing the file and using the command: python3 VectorizationAndCNN.py.

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

<p align="center">
 <img src="./EmbeddingsArray.png" width="1000" height="200">
</p>

 The second tool that goes right along with the first one is the ``index mapping``. This tool allows for the word in the tweets to be converted into a single dimensional numerical vector that acts and the intermediary between words and word embeddings. It contains the index location for every word that is in the Embeddings Array. For example, the word representation for ``Dogs`` might be stored in the second line in the ``Embeddings Array``. So in the ``Index Mapping`` the word ``Dogs`` would be matched with the number 2, representing its storage location in ``Embeddings Array``. A snipped of this mapping is shown below:
  
<p align="center">
 <img src="./IndexMapping.png" width="300" height="500">
</p>

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

<p align="center">
 <img src="./DataSplit.png" width="600" height="600">
</p>

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

# Discussion

We believe our accuracy over the test set could be greatly improved. A major cause of this is that the code does not include a dropout layer in the CNN. The result is that the CNN overfits to the training set and begins to lose accuracy on our validation set after multiple epochs.

The ablation study shows that our accuracy improves as the size of the training set increases. However, the gain in accuracy as well as the other scores is brought about by sacrificing run time and memory. An attempt to train our model for 15 epochs over the whole training set resulted in failure as Google Colab ran out of resources. Google Colab Pro was used afterwards, and the number of epochs was reduced to only 10 to resolve this problem.


# Future Work

Given unlimited time and resources, we would expand this project in a few ways. First, we would include early stopping to avoid overfitting. The keras library which was used for the CNN has a convenient dropout function. It does more than just stop the training when accuracy on the validation set begins to drop. It probabilistically removes, or “drops out,” inputs to a layer, which may be input variables in the data sample or activations from a previous layer.

The second major expansion to this project would be to use new data. A simple framework called twitter_api_framework.py in this directory provides a way to pull tweets from twitter’s API. These tweets can be requested with a query such that retrieved tweets contain the query. Other filters are possible, but the framework needs expansion. Once that is done, we could use the framework to perform sentiment analysis on tweets relating to specific topics. The most difficult issue with pulling tweets from the Twitter API would be labelling the data. We would likely have to manually label the sentiment of new tweets.
