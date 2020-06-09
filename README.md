# Self-Multi-Head-Attention-Based-CNNs-for-fake-News-Detection
An NLP Task aimed at Predicting whether a given news article is fake or real by creating own Word2Vec Model, leveraging Attention-Mechanism and then using CNNs

## Directory Structure :-

This Directory Consists for the following files :-

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

### Intuition for CNN Architecture

The Project aims at Detecting an Object using a set of points whose values are Light Intensities. This is Similar to Classifying Images which consist of Pixels and Collection of Pixels gives us the image. Finding out spatial relation/pattern among pixels helps us in Classification. Similarly here, finding spatial Pattern between light Intensities treating them as pixels can help us correctly Detect Exo-Planets by taking into consideration the relation different intensities have with each other.

### CNN Architecture Used

Among all the Architectures used for Experimentation, the most promising results were obtained from the following:-

**1. Customised AlexNet + Dense Layers** :- The results Obtained from this architecture were as follows -

| Label                 | Precision         | Recall            | f1-Score          |
| -------------         |:-----------------:|:-----------------:|:-----------------:|
| 0(Non-ExoPlanet)      | 1.00              | 0.97              | 0.98              |
| 1(Exo-Planet)         | 0.21              | 1.00              | 0.34              |

**Confusion Matrix**

| Label                 | 0(Non-ExoPlanets) | 1(Exo-Planet)    |
| -------------         |:-----------------:|:-----------------:
| 0(Non-ExoPlanets)     | 546               | 19               |
| 1(Exo-Planet)         | 0                 | 5                |

**Training and Validation Loss and Accuracy Curves**

![](Graphs/Training_And_Validation_Accuracy_AlexNet.PNG)
![](Graphs/Training_And_validation_loss_AlexNet.PNG)

**Take Away**

The test Set had 565 Non-ExoPlanets and 5 Exo-Planets. Still, the above model identified all Exo-Planets Correctly without Mis-Classification. Out of 565 Non-ExoPlanets it identified 546 Correctly. Hence, we can conclude that the above model succeeds in correctly identifying Exo-Planets, although there is error in classifying Non-ExoPlanets.

**2. Multi-Channel CNN** :-

**Intuition** :- Usually in Object Detection Tasks where Features of a good number of Negative samples match with features of Positive samples, classifiers fail to create a bias and we end up with high False Positives. To prevent this, we need to Build an architecture which can Identify Different Features of Positive Examples Separately and then combine all the features to differentiate it from Negative Examples. 

Using a Multi-Channel CNN helps here. In the first Channel, a filter of length 2001 has been used to Identify Global Features, i.e Overall Shape of Transits. In the Second Channel a filter of Length 201 has been used to Identify Local Features of Transits , i.e to make sure they are in accurate transit shape and are not Noises arising from Other Phenomenon. Finally both these Features are combined to make a prediction.

The results are as Follows -


| Label                 | Precision         | Recall            | f1-Score          |
| -------------         |:-----------------:|:-----------------:|:-----------------:|
| 0(Non-ExoPlanet)      | 1.00              | 1.00              | 1.00              |
| 1(Exo-Planet)         | 1.00              | 0.60              | 0.75              |

**Confusion Matrix**

| Label                 | 0(Non-ExoPlanets) | 1(Exo-Planet)    |
| -------------         |:-----------------:|:-----------------:
| 0(Non-ExoPlanets)     | 565               | 0                |
| 1(Exo-Planet)         | 2                 | 3                |

**Training and Validation Loss and Accuracy Curves**

![](Graphs/Training_And_validation_Accuracy_Multi-Channel.PNG)
![](Graphs/Training_And_validation_Loss_Multi-Channel.PNG)

**Take Away**

The test Set had 565 Non-ExoPlanets and 5 Exo-Planets. The  above model identified all Non-ExoPlanets Correctly without Mis-Classification. Out of 5 ExoPlanets it identified 3 Correctly. Hence, we can conclude that the above model succeeds in correctly identifying Non-ExoPlanets, although there is error in classifying Exo-Planets.

## Conclusion

The First Model is Good at Detecting Exo-Planets at Cost of Non-ExoPlanets. The Second Model Detects Non-ExoPlanets Accurately. Hence we can combine the two models together to make a reasonable prediction. When both the models predict something as Exo-Planet/Non-ExoPlanet, we can be fairly sure of our Prediction(Since Models with opposite bias agree on a prediction).


