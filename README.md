# Age and Gender Estimation

## Problem Statement

Given an image of a person, determine the gender(Male/Female) and age group(0 to 14 years, 14 to 25 years, 25 to 40 years, 40 to 60 years and more than 60 years) of a person.

## Dataset Used

The dataset is UTKFace dataset taken from Kaggle. It can be accessed [here](https://www.kaggle.com/jangedoo/utkface-new). The dataset consists of 20,000 face images with annotations of age, gender and ethnicity. However, the ethnicity has no relevance with respect to our problem statement.

## Approach

A Multi-Task CNN is used to simultaneously output the probabilities of gender and age groups. The model