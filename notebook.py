#!/usr/bin/env python
# coding: utf-8

# # **Proyek Pertama : Membuat Model NLP dengan TensorFlow**

# * Nama : Firman Nurcahyo
# * Email : firman.cahyo.369@gmail.com
# * ID DiCoding : Firman Nurcahyo

# # **Mengimport Modul Yang Diperlukan**

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Read the CSV file into a DataFrame
news_data = pd.read_csv('bbc-news-data.csv', sep='\t')

# Display the first few rows of the DataFrame
print(news_data.head())


# In[3]:


# Display information about the DataFrame
print(news_data.info())

# Display the count of unique values in the 'category' column
category_counts = news_data['category'].value_counts()
print(category_counts)


# In[4]:


# Drop the 'filename' column from the DataFrame
news_data = news_data.drop(columns=['filename'])
news_data


# In[5]:


# Set the seaborn style to "darkgrid"
sns.set_style("darkgrid")

# Create a new figure with a specified size
plt.figure(figsize=(10, 6))

# Use seaborn to create a count plot based on the 'category' column
sns.countplot(x='category', data=news_data)

# Set the title of the plot
plt.title('Distribution of Categories')

# Display the plot
plt.show()


# In[6]:


# Concatenate 'title' and 'content' columns into a new 'text' column
news_data['text'] = news_data['title'] + " " + news_data['content']
news_data


# # **Data Cleaning**

# In[7]:


# Import necessary libraries for NLP and machine learning
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import re
import string
import unicodedata
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
from keras.preprocessing import text, sequence
import nltk

# Download stopwords data from NLTK
nltk.download('stopwords')


# In[8]:


# Create a set of English stopwords
stopwords_set = set(stopwords.words('english'))

# Create a list of punctuation
punctuation_list = list(string.punctuation)

# Update the stopwords set to include punctuation
stopwords_set.update(punctuation_list)


# In[9]:


# Define and initialize stopwords set
stwd = set(stopwords.words('english'))

# Function to strip HTML tags from text
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Function to remove text between square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Function to remove URLs from text
def remove_url(text):
    return re.sub(r'http\S+', '', text)

# Function to remove stopwords from text
def remove_stopwords(text):
    final_text = []
    for word in text.split():
        if word.strip().lower() not in stwd:
            final_text.append(word.strip())
    return " ".join(final_text)

# Function to denoise text using the defined cleaning functions
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_url(text)
    text = remove_stopwords(text)
    return text

# Apply the denoise_text function to the 'text' column of the DataFrame
news_data['text'] = news_data['text'].apply(denoise_text)


# In[10]:


# Function to get the corpus (list of words) from a list of texts
def get_corpus(text):
    words = []
    for sentence in text:
        for word in sentence.split():
            words.append(word.strip())
    return words

# Apply the get_corpus function to the 'text' column of the DataFrame
corpus = get_corpus(news_data['text'])

# Display the first 10 words in the corpus
print(corpus[:10])


# In[11]:


# Import Counter from the collections module
from collections import Counter

# Display the dictionary of the 10 most common words
counter = Counter(corpus)
most_common = dict(counter.most_common(10))
most_common


# In[12]:


# Function to get the top text n-grams
def get_top_text_ngrams(corpus, n, g):
    cv = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_words = cv.transform(corpus)
    sum_words = bag_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Create a bar plot of the top 10 unigrams
plt.figure(figsize=(16, 9))
most_common = get_top_text_ngrams(news_data['text'], 10, 1)
most_common = dict(most_common)
sns.barplot(x=list(most_common.values()), y=list(most_common.keys()))
plt.show()


# In[13]:


# One-hot encode the 'category' column
category = pd.get_dummies(news_data['category'])

# Concatenate the original DataFrame with the dummy variables
new_cat = pd.concat([news_data, category], axis=1)

# Drop the original 'category' column
new_cat = new_cat.drop(columns='category')

# Display the first 10 rows of the modified DataFrame
new_cat.head(10)


# In[14]:


# Extract 'text' and 'category' columns
news = new_cat['text'].values
label = new_cat[['business', 'entertainment', 'politics', 'sport', 'tech']].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(news, label, test_size=0.2, shuffle=True)


# # **Tokenizer Tensorflow**

# In[15]:


# Import TensorFlow and necessary modules
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# In[16]:


# Set parameters for tokenization and padding
vocab_size = 10000
max_len = 200
trunc_type = "post"
oov_tok = "<OOV>"

# Create a Tokenizer with specified parameters
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# Fit the Tokenizer on the training data
tokenizer.fit_on_texts(x_train)

# Get the word index from the Tokenizer
word_index = tokenizer.word_index

# Convert text data to sequences
sequences_train = tokenizer.texts_to_sequences(x_train)
sequences_test = tokenizer.texts_to_sequences(x_test)

# Pad sequences to a specified length
pad_train = pad_sequences(sequences_train, maxlen=max_len, truncating=trunc_type)
pad_test = pad_sequences(sequences_test, maxlen=max_len, truncating=trunc_type)

# Display the shape of the padded test data
print(pad_test.shape)


# In[17]:


# Display the padded training and testing data
print("Padded Training Data")
print(pad_train)

print("\nPadded Testing Data")
print(pad_test)


# In[18]:


# Define the sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile the model with specified optimizer, metrics, and loss function
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

# Display a summary of the model architecture
model.summary()


# In[19]:


# Define a custom callback class
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # Check if both training and validation accuracy are greater than 93%
        if(logs.get('accuracy') > 0.93 and logs.get('val_accuracy') > 0.93):
            # Stop training if the condition is met
            self.model.stop_training = True
            print("\nAccuracy for both training set and the validation set has reached > 93%!")

# Instantiate an object of the custom callback class
callbacks = myCallback()


# In[20]:


# Specify the number of epochs
num_epochs = 50

# Train the model with the provided data
history = model.fit(
    pad_train,
    y_train,
    epochs=num_epochs,
    validation_data=(pad_test, y_test),
    verbose=2,
    callbacks=[callbacks]
)


# In[21]:


# Plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()


# In[22]:


# Plot the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

