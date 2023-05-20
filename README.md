# ProjectNLP_2023


<a name="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/s26584/ProjectNLP_2023">
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" fill="currentColor" class="bi bi-robot" viewBox="0 0 16 16">
  <path d="M6 12.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 0 1h-3a.5.5 0 0 1-.5-.5ZM3 8.062C3 6.76 4.235 5.765 5.53 5.886a26.58 26.58 0 0 0 4.94 0C11.765 5.765 13 6.76 13 8.062v1.157a.933.933 0 0 1-.765.935c-.845.147-2.34.346-4.235.346-1.895 0-3.39-.2-4.235-.346A.933.933 0 0 1 3 9.219V8.062Zm4.542-.827a.25.25 0 0 0-.217.068l-.92.9a24.767 24.767 0 0 1-1.871-.183.25.25 0 0 0-.068.495c.55.076 1.232.149 2.02.193a.25.25 0 0 0 .189-.071l.754-.736.847 1.71a.25.25 0 0 0 .404.062l.932-.97a25.286 25.286 0 0 0 1.922-.188.25.25 0 0 0-.068-.495c-.538.074-1.207.145-1.98.189a.25.25 0 0 0-.166.076l-.754.785-.842-1.7a.25.25 0 0 0-.182-.135Z"/>
  <path d="M8.5 1.866a1 1 0 1 0-1 0V3h-2A4.5 4.5 0 0 0 1 7.5V8a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1v1a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-1a1 1 0 0 0 1-1V9a1 1 0 0 0-1-1v-.5A4.5 4.5 0 0 0 10.5 3h-2V1.866ZM14 7.5V13a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V7.5A3.5 3.5 0 0 1 5.5 4h5A3.5 3.5 0 0 1 14 7.5Z"/>
</svg>
  </a>

  <h3 align="center">NLP Project 2023</h3>
  <p align="center">
    Twitter Sentiment Analysis Using Python
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-verview">Project Overview</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#result">Result</a></li>
    <li><a href="#github-repository">Github repository</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Project Overview

The aim of this project is to create machine larning models for sentiment analysis based on tweets about the war in Ukraine.

The dataset used for this project consists of 200,000 English tweets posted by random Twitter users on the 30th of March, 2023.
Before building machine learning models, the original dataset (`tweets_en.csv`) was cleaned and lablelled following the procedure below.

Data Preparation:
* Set the number of target classes as 2 (positive and negative)
* Remove stopwords, special characters, URL, any unecessary contents for sentiment analysis.
* Create word embeddings
* Create clusters using K-MEANS

After cleaning and labelling the dataset (`labeled_dataset.csv`), 5 types of machine learning models were build. The dataset is unbalanced (positive:negative = 7:1), hence precision, recall and F1 score were used to measure the performance.

Machine Learning Models:
* Classic machine learning
  <ul>
    <li><b>Logistic regression</b></li>
    <li><b>SVM - Linear SVC</b></li>
    <li><b>Bernoulli Naive Bayes</b></li>
  </ul>
* <b>Convolutional neural network (CNN)</b>
* <b>BERT language model</b>


## Built With

Below is the list of frameworks/libraries used to build models.

* Logistic regression by <i>scikit-learn</i>: [official documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
* Linear SVC by <i>scikit-learn</i>: [official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
* Bernoulli Naive Bayes by <i>scikit-learn</i>: [official documentation](https://scikit-learn.org/stable/modules/naive_bayes.html#bernoulli-naive-bayes)
* 1D convolution by <i>Keras</i> from <i>TensorFlow</i>: [official documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D)
* BERT by <i>Google</i>: [official documentation](https://huggingface.co/docs/transformers/model_doc/bert)


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Below is the required packages to be installed.
  - `unidecode`
  - `plot_keras_history`
  - `transformers`
  - `datasets`
  - `accelerate`

<!-- USAGE EXAMPLES -->
## Usage

- To use your Google Drive, please customize the path name <i>'drive/MyDrive/ZUM/project1/'</i> in the code.
- To use the saved models, please load files from /saved_models/ in this project repo.
- `tweets_en.csv` and `labeled_dataset.csv` in /dataset/ are compressed as zip and 7z file respectively. Please extract them before using.
- <b>WARNING</b>: The original dataset was created by using `snscrape` in March, 2023. However, <b>snscrape for Twitter is not available since April, 2023</b>. To train and test your model, please use the datsets in /dataset/ in this project repo.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/s26584/ProjectNLP_2023.git
   ```
2. Install necessary packages
   ```sh
   !pip install snscrape
   !pip install unidecode
   !pip install plot_keras_history
   !pip install -U transformers
   !pip install -U datasets
   !pip install --upgrade accelerate

   # Run below code in case TrainingArguments causes following error:
   # NameError: name 'PartialState' is not defined in TrainingArguments
   !pip uninstall -y transformers accelerate
   !pip install transformers accelerate
   ```
### Example of loading saved models
* Pickle file
  ```py
  from sklearn.metrics import f1_score

  filename = 'your_path_here/bernoulli_model.pkl'
  loaded_model = pickle.load(open(filename, 'rb'))
  f1_value = f1_score(y_test, loaded_model.predict(X_test), average='macro')
  print(f1_value)
  ```
* JSON and HDF5 files
  ```py
  from tensorflow.keras.models import model_from_json
  
  filename = 'your_path_here/model_cnn.json'
  json_file = open(filename, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  
  # load weights into new model
  weigth = 'your_path_here/model_cnn.h5'
  loaded_model.load_weights(weigth)

  precision = keras.metrics.Precision()
  recall = keras.metrics.Recall()

  # evaluate loaded model on test data
  loaded_model.compile(loss='categorical_crossentropy', 
                        optimizer='rmsprop', 
                        metrics=[precision, recall])
                        
  loss, precision, recall = loaded_model.evaluate(X_test, y_test)
  print(f"Test loss: {loss:.3f}") 
  print(f"Test precision: {precision:.3f}")
  print(f"Test recall: {recall:.3f}")
  test_f1 = 2 * (precision * recall) / (precision + recall)
  print(f"Test f1 score: {test_f1:.3f}")
  ```
 

<!-- RESULT -->
## Result
The table below illustrates the test performance of each model.
*(0) and (1) indicate the class labels (0: positive, 1: negative)

|         Model       | Loss |   Precision   | Recall  | F1 Score |
|:--------------------| :----------: |:---------: |:------:| -----:|
| Logistic Regression | 4.5 (MAE), 4.5(MSE) | 96.0 (0), 92.0 (1) | 99.0 (0), 60.0 (1) | 98.0 (0), 73.0 (1) |
|     Linear SVC      | 3.4 (MAE), 3.4 (MSE) | 97.0(0), 90.0 (1) | 99.0 (0), 75.0 (1) | 98.0 (0), 82.0 (1) |
|Bernoulli Naive Bayes| 8.7 (MAE), 8.7(MES) | 91.0 (0), 96.0 (1) | 100.0 (0), 14.0 (1) | 95.0 (0), 24.0 (1) |
|         CNN         | 19.8 | 94.7 | 94.7 | 94.7 |
|         BERT        | 14.2 |  N/A |  N/A | 85.1 |


<!-- repo -->
## Github repository

Project Link: [https://github.com/s26584/ProjectNLP_2023](https://github.com/s26584/ProjectNLP_2023)



<p align="right">(<a href="#readme-top">back to top</a>)</p>

