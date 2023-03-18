# Use case
Fashion recommendation prediction based on textual review from consumers

# Data
Name: Women’s E-commerce Clothing Reviews
<br>Source: Kaggle
<br>URL: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews</br>

# Data exploration
- 23486 entries
- 10 columns
  *  Qualitative 
    1. Title is a string variable for the title of the review.
    1. Review is the review body.
    1. Division Name is the Categorical name of the product high level division.
    1. Department Name is Categorical name of the product department name.
    1. Class Name is the Categorical name of the product class name.
  -  Quantitative
    1. Clothing ID which is an integer categorical variable that refers to the specific piece being reviewed.
    1. Age of reviewers
    1. Rating is a positive ordinal Integer variable for the product score granted by the customer from worst rating 1 , to best rating 5.
    1. Positive Feedback Count is a Positive Integer documenting the number of other customers who found this review positive.
    1. Recommended IND, the target binary variable if the customer recommends the product where 1 is recommended, 0 is not recommended.

Due to computing contraints, ***a random sample without replacement*** of 6000 entries from the original data was taken.
<br>
And the target proportions are very similar between the random sample and the original data. That is, about ***82%*** and ***18%*** for ***recommended*** and ***non-recommended*** group, respectively.
![Picture1](https://user-images.githubusercontent.com/35198050/226122101-c659f63c-713a-4fb0-be1c-e6e54e533b77.png)

- Duplicates
  - 2 identified and removed
- Missing data
  - Division, department and class were imputed by “unknown”. But there were only 2.
  - Numeric columns have no missing data.

In order to address the ***class imbalance***, class weights for training set were computed as follows:
<br>For example, the class weight for “non-recommended group”, the ***numerator*** is the total number of row entries in the training set.
<br> And the ***denominator*** is the result of the number of distinct groups and in our case it is 2, multiplies by total number of entries in the non-recommended group.
![Picture8](https://user-images.githubusercontent.com/35198050/226125375-9eac41d6-254c-413c-abaa-5dcdb9c0933a.png)

# Data visualisation
Age of reviewers and positive feedback counts shows similar distribution for both groups. On the other hand, it is not the case for rating. 
<br>It is much higher for recommended review than non-recommended which is reasonable. The better the rating the more likely the recommendation.
![Picture2](https://user-images.githubusercontent.com/35198050/226125814-bb586576-2da6-4159-8ec5-60cb12f1a6c6.png)

As for correlation, we could say the correlation between the numerical features are extremely weak, they are all below the absolute value of 0.1. Therefore, they were all included in the baseline simple logistic regression.
![Picture3](https://user-images.githubusercontent.com/35198050/226125914-83dfad48-30a3-4243-a040-4224ea6dde00.png)

The stacked bar chart clearly shows rating 4 and 5 have the higher recommended proportions. Whereas, not recommended group has the most poor rating proportions. This pattern has been as well illustrated in the earlier slide on boxplot.
![Picture4](https://user-images.githubusercontent.com/35198050/226125966-bb95070a-1fef-4ba1-9191-39c5a3a0f0a5.png)
As for the Department, most reviews were on department Tops. Department Intimate and Jackets both display similar trend in terms of proportion. Department Trend has the least reviews. Why? Does it mean this department did the worst in terms of sales? Perhaps the business could look into this. Department “Unknown” was an imputed value and there were 2 as explained earlier due to missing data.
![Picture5](https://user-images.githubusercontent.com/35198050/226125971-4be8c548-baa4-41a1-a5f4-71d98754e6c5.png)
For Division, most reviews were on General, followed by General Petite and Intimates.
Unknown is due to imputation.
![Picture6](https://user-images.githubusercontent.com/35198050/226125974-64da1255-e047-4145-b8b1-062fdd77d2f0.png)
Lastly the product class name, it seems like women prefers to buy dresses the most because it has the most reviews and the least on layering. Again, the unknown is due to imputation. 
![Picture7](https://user-images.githubusercontent.com/35198050/226125976-da927db6-81d7-45b5-a5b7-1051e70d93dc.png)

# Feature engineering
- Incorporate both the text and non-text features as a single textual feature as follows. The idea comes from Chris<sup>[1](https://mccormickml.com/2021/06/29/combining-categorical-numerical-features-with-bert/)</sup>. <section>
This item comes from {"Department Name} department and division is {“Division Name”}, and is classified under {“Class Name”}. There are {"Positive Feedback Count”} customers who found this review
positive. I am {“Age”} years old. I rate this item {“Rating”} out of 5 stars.
</section>

# Classification algorithms
Simple logistic regression<sup>[2](https://spark.apache.org/docs/latest/ml-tuning.html)</sup> | Logisitc regression<sup>[2](https://spark.apache.org/docs/latest/ml-tuning.html)</sup> | Random Forest<sup>[2](https://spark.apache.org/docs/latest/ml-tuning.html)</sup> | Deep Learning
---:|:---:|:---:| --- 
**Pyspark 3.3.0** | **Pyspark 3.3.0** | **Pyspark 3.3.0** | **Spark NLP 4.0.1**<sup>[3](https://towardsdatascience.com/text-classification-in-spark-nlp-with-bert-and-universal-sentence-encoders-e644d618ca32)</sup>
Baseline model | TF-IDF | TF-IDF | Transformer-based Universal Sentence Encoder<sup>[4](https://nlp.johnsnowlabs.com/2020/04/17/tfhub_use.html), [5](https://nlp.johnsnowlabs.com/docs/en/transformers), [6](https://tfhub.dev/google/universal-sentence-encoder-large/5)</sup>
Numerical features <br>(Age, rating and Positive Feedback Count) | Grid search for hyperparameter tuning | Grid search for hyperparameter tuning |  Default hyperparameters<sup>[7](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLApproach)</sup>
3-fold cross validation | 3-fold cross validation | 3-fold cross validation |

The first three classifiers were done using Pyspark which is the Python API for Apache Spark whereas Spark NLP is the Natural Language Processing library built on top of Apache Spark. 

I started with a simple logistic regression with only the three numerical features, age, rating and positive feedback counts. Performed the hyperparameter tuning and 3-fold cross validation for the optimal hyperparameter through a pipeline before assessing the model performance. Due to resource limitation, I only did a 3-fold cross-validation. You might want to increase it if it will improve the model performance if you have the resource available but, 3-fold seems to be doing already very well given the data and resources in this case. We will look at the evaluation metric for this 4 classifiers in our next slides.

The other 2 classifiers of logistic regression and random forest, however, using Term frequency-inverse document frequency, TF-IDF for short as input feature after some text preprocessing on the consolidated text mentioned just now. It is a popular, simple and effective weighting scheme of the word representation. The text-preprocessing I performed included,
Tokenization which is the process of splitting the text into smaller units known as tokens,
Normalization,  to keep alphanumeric and remove punctuations.
Stop word removal. Stop words are words so common that they can be removed without significantly changing the meaning of a text. Which is useful when one wants to deal with only the most semantically important words in a text, and ignore words that are rarely semantically relevant, such as articles and prepositions.
Stemmer, to return the hard-stems out of words with the objective of retrieving the meaningful part of the words.

Lastly the deep learning approach, where I have used the pre-trained model called transformer-based universal sentence encoder from Spark NLP. The Universal Sentence Encoder encodes text into high-dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks. The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is variable length English text and the output is a 512 dimensional vector. I decided to use the default hyperparameter without tuning because we have already achieved good result with classical algorithms.

# Confusion Matrix
Here is the confusion matrix of the 4 classifiers. It defines the base for performance measures.\
Simple logistic regression gives the least misclassifications of only 66 i.e. 8+58.\
Followed by logistic regression and random forest  using Term frequency-inverse document frequency with 103, and 104 respectively.\
Deep learning approach is the worst where it predicts everything as recommended.
![Picture9](https://user-images.githubusercontent.com/35198050/226137025-1222bd52-2a69-47f2-bbf9-0e698f22544f.png)

# Performance Metrics
![Picture10](https://user-images.githubusercontent.com/35198050/226137591-fed63bc9-7d23-4655-bf3a-4f39eed7418e.png)
As we have class imbalance, I would only look at F1-score and area under the precision-recall curve because they are more robust evaluation metrics which is less likely to be biased to the majority or minority class.
That means, simple logistic regression is the best according to the evaluation metrics and training time. However, it uses information of only the numerical features.
Deep learning is the worst. The result is reflected in the earlier confusion matrix where it fails to predict the non-recommended.
Logistic regression and random forest with Term frequency-inverse document frequency are comparable in terms of evaluation metrics. However, random forest took the longest time of 2.5hours to train whereas it’s only about 7 mins for logistic regression.
Therefore, logistic regression with Term frequency-inverse document frequency is recommended. 

# Conclusion
To conclude, studying online reviews with machine learning models allows businesses to develop their services with ideas and to meet consumer satisfaction through their buying trends and behavior.

# References
1. https://mccormickml.com/2021/06/29/combining-categorical-numerical-features-with-bert/
1. https://spark.apache.org/docs/latest/ml-tuning.html
1. https://towardsdatascience.com/text-classification-in-spark-nlp-with-bert-and-universal-sentence-encoders-e644d618ca32
1. https://nlp.johnsnowlabs.com/2020/04/17/tfhub_use.html
1. https://nlp.johnsnowlabs.com/docs/en/transformers
1. https://tfhub.dev/google/universal-sentence-encoder-large/5
1. https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLApproach
1. https://www.coursera.org/learn/advanced-data-science-capstone/home/assignments
