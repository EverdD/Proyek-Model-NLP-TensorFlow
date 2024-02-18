# Natural Language Processing (NLP) Model with TensorFlow

This project focuses on developing a Natural Language Processing (NLP) model using TensorFlow. The model is trained on a dataset containing news articles from different categories such as business, entertainment, politics, sport, and technology. The goal is to classify news articles into their respective categories based on their textual content.

## Features

- Utilizes LSTM (Long Short-Term Memory) neural network architecture for sequence prediction.
- Implements tokenization and padding of text data for model training.
- Cleans and preprocesses textual data using techniques such as HTML tag removal, URL removal, and stopword removal.
- Visualizes model performance using training and validation accuracy/loss curves.
- Provides a custom callback to stop training when desired performance is achieved.

## Dataset

The dataset used in this project can be downloaded from the following link:

[Download Dataset](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive/download?datasetVersionNumber=1)

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- NLTK
- BeautifulSoup

## How to Run

1. Download the dataset from the provided link and save it to the project directory.
2. Install all necessary dependencies by running `pip install -r requirements.txt`.
3. Run the Python script in an environment that supports TensorFlow.
4. Follow the instructions in the script to preprocess the data, train the model, and evaluate its performance.

## Usage

- Train your own NLP model using the provided script.
- Visualize the training and validation accuracy/loss curves to monitor model performance.
- Test the model's predictions by providing new textual data.
