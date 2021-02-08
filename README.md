# Read Me File For Text Classification Project

## Problem Statement
Take five different samples of Gutenberg digital books , which are of five different genres and authors, that are semantically different. Separate and set aside unbiased random partitions for training, validation and testing.

## Abstract
We take five different samples of Gutenberg digital books, which is a is a large electronic collection of over 54,000 public domain books. We Separate and set aside unbiased random partitions for training, validation and testing.

### Goal
The overall objective is to produce classification predictions and compare them; analyze the pros and cons of algorithms and generate and communicate the insights.
Gauge the bias and variability of the models to decide the champion model.

### Hypothesis
Our hypothesis is that each author uses a certain set of words, and we aim to train the machine to learn this and classify texts by author.

We work on the four different algorithms namely, SVC (Support Vector Classifier), Random Forests, Naive Bayes and K Nearest Neighbours with different pre-processing techniques like - TF-IDF and Bag of Words and then chose our champion model having the highest accuracy. 

# Project Setup
1. Load Google colab - https://colab.research.google.com/
2. Create a new notebook 
3. Upload the xlsx file in the file section 

### Note
The xlsx file that you upload will fanish if the session expires. 
If uploading the file from local remember to have the file in the folder where the .py file is located. 
Exact path should be mentioned along with the file name in order to upload the file correctly
Order of the cells should be followed during execution for optimzed output and avoiding the re-runs. 

# Process
Import all the required libraries. 

## Step 1: Data
Function to return partitioned book as a dataframe with 200 rows , each row containing 100 words
1) We use the word_tokenize package from nltk to tokenize the package
2) Using a while loop, we partition the book until the end of passage :
We grab 100 of the next token at a time
Using regex, we then calculate the number of tokens (n) out of the 100 that are not words but rather punctuations etc.
However, it is possible that the next(n) tokens also have some punctuations. So we look ahead and once again we calculate the number of tokens(n2) that are not words
We repeat this to get the index (100+n+n2+..) that would allow for 100 words in the passage.
This passage is then added onto the BookDf dataframe.
We repeat this process from the previous index counter.
3) If the dataframe has less than 200 rows, this means that there were not enough number of words in the book (<20000) for 200 partitions of 100 words each. This function will still return a dataframe back but gives a warning.
4) If the total length has greater than 200 rows, we used numpy's random number generator to get200 random non-repeating numbers between 0 and max number of book rows. We then return this book back to the user.

**The book partition is done into 200 random partitions of 100 words each.**

We can see our partitions by calling df1

### Note
Please make sure book.xlsx is in the proper location and change following path to correspond to the file location.
Using the returnLabeledBook would generate a different combination of training/testing dataset each time. This could impact the analysis results.
In order to retain consistent results each time the notebook is rerun, we are loading the data set from books.xlsx excel file.
If there is an error in loading data from this excel, then different results may not match he documentation due to stochastic randomness in grouping the training/testing set.

## Data Processing and Cleansing
Getting rid of the punctuations and Spacy Lemmatization

Our choice of Spacy libary was motivated by the fact that it can perform tokenization in 0.2 milliseconds compared to nltk's 4 milliseconds.

We are performing a lemmatization in the code. It is a process of converting many different forms to it's root word. 

The nlp pipeline created using Spacy will automatically perform the tokenization, parsing and tagging processes for us.

We have reran this code without these preprocessing step which lead to lower accuracy.

We are not removing stopwords during preprocessing because these features assists in adding context to the classification. Because our data contains past century books such as Shakespeare's Hamlet, the common language are different then than modern books. Thhese stopwords could serve as key feature to help distinguish these differences. Removing them could potentially reduce the accuracy of the model in classifying such books.

Then perform the Training/Testing Data Split on the book partitions dataframe using the following 

## Step 2 : Feature Engineering

As a first choice, we are using TF-IDF to transform our X (Passages) into vectors

TF-IDF is a vectorization algorithm which is used to represent textual data in numerical vectors . It gives weights to the words depending on their frequency.
Compared to bag of words, TF-IDF also calculates the inverse document frequency, which will factor in the frequency of the word to occur in all documents. This will take out very commonly used words.
The 'Fit' is used to identify the vocabulary and frequency whereas the 'Transform' is used for conversion to a vector.

## Step 3 : Model Selection

**SVC Algorithm with TF-IDF**
The purpose of a Linear SVC i.e Support Vector Classifier is to return a hyperplane that seperates the data into groups.

**Hyper parameter tuning**
We are performing hyper parameter tuning on
C
type of loss

**Grid search and 10 fold cross validation**
We use a grid search to identify the best combination of the hyper parameters that provides the best accuracy results
During this grid search we try each of the parameter combination on a 10-fold cross validation
In cross validation we are splitting the training data again into 10 seperate parts, and holding one part out for testing each time.

The following function concatenates the grid_search results into a more readable dataframe with the below results
test scores of CV1,CV2,CV3,CV4,CV5,CV6,CV7,CV8,CV9,CV10
train scores of CV1,CV2,CV3,CV4,CV5,CV6,CV7,CV8,CV9,CV10
average train score from 10 fold cross validation
average test score from 10 fold cross validation
The corresponding parameter combinations for these features


Grid search results returned for Tf-idf and SVC algorithm.
All results are sorted by the worse to best test scores averaged from the 10-Fold Cross Validation. In some instances, multiple records can have equal (best) score. In such case, we are selecting the first best combination we come across, because any parameter combinations with yield the same test scores.

**Learning Curve**
These are graphic representations for selecting the best parameter combination. We look to see whether increasing a parameter value will yield overfitting.

We plot a learning curve using the following code

**Implementation of the Tf-idf and SVC algorithm with the optimal hyper parameters determined from grid search on the test data.**

We train the model using the vectorized X train data and Y train. We predict the results using the vectorized X test data.
This tuned model will be used to predict the true test/holdout data.

**Testing Results**
Confusion matrix and Classification report for the Tf-idf/SVC(C=3,loss='squared_hinge') classifier's performance

**ROC Curve**
A Receiver Operating Characteristic curve is a graph that shows the performance of a classification model for different classes based on the true positive and false positive rates.

The areas under the curve is high (actually perfect with 1.0) for Shakespeare-Hamlet and Milton-paradise, demonstrating that the model can always correctly classify 16th century classics. The area under the curve is a lower for other modern books ( chesterton-thursday with 0.99 and byrant-stories with 0,98), showing that it can occasionally misclassify modern writing styles.

We plot an ROC curve using the accuracy generated by the classification report


**Error Analysis**
Error Analysis is used to determine the type of errors made by the model and identify the causes for such a misclassification, in order to define an approach to fix them in the future.

The following is the list of misclassified labels


We then Use the Wordclouds for visualizing the type of words used in the passages that were misclassified to identify errors


We use Unigrams and Bigrams see the words that occur most frequently - alone(unigram) or together in pairs of two(bigrams)

**We Repeat the same process for all combinations of the following algorithms - 
SVC/Bag of Words
Random Forest/TF-IDF
Random Forest/Bag of Words
Naive Bayes/TF-IDF
Naive Bayes/Bag of Words
KNN/TF-IDF
KNN/Bag of Words**

**We compare the accuracies of all the models and chose our champion model**

In addition to having the best model, we try to manipulate the values of the words in the passage in order to make it harder for the model to predict. 

**Plotting a graph for the Passage words vs Test Accuracy**
This plot shows that with increasing words in the passage, we are able to predict the corresponding book with higher accuracy . However, this is a strong classifier, as it is able to classify the test datset of 40 words and greater at an accuracy of 90% and greater.

