# Self-Multi-Head-Attention-Based-CNNs-for-fake-News-Detection
An NLP Task aimed at Predicting whether a given news article is fake or real by creating own Word2Vec Model, leveraging Attention-Mechanism and then using CNNs

## Directory Structure :-

This Directory Consists of the following files :-

**1.NoteBooks Training Word2Vec** :- This folder Contains a Jupyter Notebook on training a customised Word2Vec Model using Fake News Dataset. The Word-Embedding Dimension and Word-Frequency-Count can be changed for Experimenation.

**2.NoteBooks Fake News Prediction** :- This folder contains Notebooks on various experiments performed for Fake News Prediction, for eg. Using Custom Trained Word2Vec + Attention Model + CNNs , Using Glove Embedding + CNNs etc. Results from all experiments are summarised in the readme file.

**3.w2v_model** :- Custom Word2Vec Model trained with minimum word frequency 5.

**4.w2v_model_New** :- Custom Word2Vec Model trained with minimum word frequency 2.

**5.Graphs** :- This Folder Contains Training and Validation Accuracy and Training and Validation Loss Graphs for the various Architectures used for Experiments.

**6.Custom_Word2Vec+Attention+CNNs.h5** :- Saved weights for the Attention Model. The JSON file for this Architecture is also given.

## Project Summary

### Data Pre-Processing

#### Tokenization and Padding

The first Step of Any NLP Classification Task is to Tokenize all words(or character/sentences depending upon which choice is more appropriate). After Tokenization, to convert the data into a matrix, all sequences must to padded to a common length. Usually, the Padding length is the longest length in the Sequences. In this Project However, Padding with Longest Sequence resulted in huge number of Sparse Vectors(lots of 0's) and the models trained on such data gave poor results. Hence, a padding length was decided based on:-

**a.** The number of Articles which had length less than the selected length.

**b.** The Model's Performance when it was trained with this padding length.

Taking the above two factors into Consideration, the two most suitable lengths were found experimentally to be 500 and 1500.

#### Detecting Fake News from Spam Level

The Spam Scores in the dataset showed the percentage of Spam Content in the Article. Any article with Spam Content more than 20 % were marked as fake. This bias level was decided by Experimentation.

#### Over-Sampling

It was Observed that after building dataset using 20 % spam level, it was Highly Imbalanced. It contained around 97 % Data Points for Real News Articles  and Just 3 % Data Points for Fake Articles. Hence, the dataset was Split Into Training and Validation Sets, after which the Training Set was Over Sampled Using SMOTE(Synthetic Minority Over Sampling Technique) which doesn't duplicate Minority Class data points rather creates Synthetic Examples Similar to the ones present in the dataset. Since only the Training Set was Over Sampled, there was no risk of Data Leakage into Validation Set.

### Word Vectorization

To find Relationship between words easily for accomplishing an NLP task, words are converted into Vectors of certain dimensions. Two Popular Choices for Converting Words to Vectors are **Word2Vec** method and **GLOVE Word Embeddings**. A custom Word2Vec Model was trained using the corpus in Fake News Dataset. Code for Custom Word2Vec model can be found [here](https://github.com/omkaranustoop/Self-Multi-Head-Attention-Based-CNNs-for-fake-News-Detection/tree/master/NoteBooks%20Training%20Word2Vec). Dimension for Word-Embedding was Selected to be 100 based on Experiment Results. The trained Word2Vec Files can be found in the Directory. These trained word Vectors were used in the Embedding Layer in the Keras Model.

### Attention Layer

A Custom Attention Layer was built in Keras by building an Attention Class. The word Embeddings obtained from Word2Vec File were fed to this Attention layer to generate a Combined Vector Matrix. This Matrix was then fed to CNNs.

### Experiment Results

Results Obtained from Different Architectures are Summarised as follows :-

**1. Custom Word2Vec + Customised AlexNet** :- The results Obtained from this architecture were as follows -

| Label                 | Precision         | Recall            | f1-Score          |
| -------------         |:-----------------:|:-----------------:|:-----------------:|
| 0(Real Article)       | 0.97              | 0.99              | 0.98              |
| 1(Fake Article)       | 0.45              | 0.22              | 0.29              |

**Confusion Matrix**

| Label                 | 0(Real Article)   | 1(Fake Article)  |
| -------------         |:-----------------:|:-----------------:
| 0(Real Article)       | 2321              | 28               |
| 1(Fake Article)       | 83                | 23               |

**Training and Validation Loss and Accuracy Curves**

![](Graphs/Custom%20Word2Vec%20%2B%20Customised%20AlexNet/Accuracy.PNG)
![](Graphs/Custom%20Word2Vec%20%2B%20Customised%20AlexNet/Loss.PNG)

**Take Away**

The test Set had 2404 Real Articles and 51 Fake Articles. Still, the above model identified 2321/2404 Real Articles Correctly. Out of 51 Fake Articles it identified 23 Correctly. Hence, we can conclude that the above model is good at correctly identifying Real Arcticles, whereas it's performance is average at Identifying Fake Articles.

**2. Custom Word2Vec + CNNs** :-

The results are as Follows -


| Label                 | Precision         | Recall            | f1-Score          |
| -------------         |:-----------------:|:-----------------:|:-----------------:|
| 0(Real Article)       | 0.96              | 0.98              | 0.97              |
| 1(Fake Article)       | 0.29              | 0.19              | 0.23              |

**Confusion Matrix**

| Label                 | 0(Real Article)   | 1(Fake Article)  |
| -------------         |:-----------------:|:-----------------:
| 0(Real Article)       | 2301              | 48               |
| 1(Fake Article)       | 86                | 20               |

**Training and Validation Loss and Accuracy Curves**

![](Graphs/Custom%20Word2Vec%20%2B%20CNNs/Accuracy.PNG)
![](Graphs/Custom%20Word2Vec%20%2B%20CNNs/Loss.PNG)

**Take Away**

The test Set had 2387 Real Articles and 68 Fake Articles. The above model identified 2301/2387 Real Articles Correctly. Out of 68 Fake Articles it identified 20 as Fake. Hence, we can conclude that the above model is better at correctly identifying Real Arcticles than the previous Model, whereas it's performance is worse at Identifying Fake Articles.


**3. Glove Embedding + CNNs** :-

The results are as Follows -


| Label                 | Precision         | Recall            | f1-Score          |
| -------------         |:-----------------:|:-----------------:|:-----------------:|
| 0(Real Article)       | 0.99              | 0.96              | 0.97              |
| 1(Fake Article)       | 0.01              | 0.25              | 0.01              |

**Confusion Matrix**

| Label                 | 0(Real Article)   | 1(Fake Article)  |
| -------------         |:-----------------:|:-----------------:
| 0(Real Article)       | 2357              | 3                |
| 1(Fake Article)       | 94                | 1                |

**Training and Validation Loss and Accuracy Curves**

![](Graphs/Glove%20%2B%20CNNs/Accuracy.PNG)
![](Graphs/Glove%20%2B%20CNNs/Loss.PNG)

**Take Away**

The test Set had 2451 Real Articles and 4 Fake Articles. The above model identified 2357/2451 Real Articles Correctly. Out of 4 Fake Articles it identified 1 as Fake. Hence, we can conclude that the above model is so far the best at correctly identifying Real Articles, whereas it's performance is relatively the worst at Identifying Fake Articles.

**Custom Word2Vec + Attention + CNNs** :-

The results are as Follows -


| Label                 | Precision         | Recall            | f1-Score          |
| -------------         |:-----------------:|:-----------------:|:-----------------:|
| 0(Real Article)       | 0.97              | 0.95              | 0.96              |
| 1(Fake Article)       | 0.19              | 0.30              | 0.23              |

**Confusion Matrix**

| Label                 | 0(Real Article)   | 1(Fake Article)  |
| -------------         |:-----------------:|:-----------------:
| 0(Real Article)       | 2228              | 126              |
| 1(Fake Article)       | 71                | 30               |

**Training and Validation Loss and Accuracy Curves**

![](Graphs/Custom%20Word2Vec%20%2B%20Customised%20AlexNet/Accuracy.PNG)
![](Graphs/Custom%20Word2Vec%20%2B%20Customised%20AlexNet/Loss.PNG)

**Take Away**

The test Set had 2299 Real Articles and 156 Fake Articles. The above model identified 2228/2299 Real Articles Correctly. Out of 156 Fake Articles it identified 30 as Fake. Hence, we can conclude that the above model is the best among all models at correctly identifying Real Arcticles, whereas it's performance is 3rd best at Identifying Fake Articles(The first two models have relatively better Performance).

## Conclusion

Adding attention layer helps in Improving detection of Real Articles. However, it results in Mis-Classification of Fake Articles as Real Ones.


