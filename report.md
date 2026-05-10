# Assignment 1: Movie Review Sentiment Classification using Embeddings

**Author:** Tanjila Medha  
**Date:** May 10, 2026

## Problem Definition
The goal of this project is to classify movie reviews as positive or negative 
using text embeddings. Sentiment analysis is a fundamental challenge in natural 
language processing with applications in recommendation systems, content 
moderation, and market research.

## Dataset
The base dataset is the IMDB movie review dataset, which was processed and 
filtered to create a custom dataset. Reviews were filtered to a maximum length 
of 1000 characters to focus on concise, punchy reviews, and a text_length 
feature column was added. The final dataset contains balanced positive and 
negative samples across train and test splits.

- **Hugging Face Dataset:** https://huggingface.co/datasets/medhuuu/movie-sentiment-dataset

## Approach
Text embeddings were generated using the `all-MiniLM-L6-v2` sentence transformer 
model. This model converts each review into a 384-dimensional vector where 
semantically similar texts are placed close together in vector space. These 
embedding vectors were then used as input features for three classifiers.

**Embedding type:** Sentence embeddings (dense, semantic)  
**Embedding model:** all-MiniLM-L6-v2 (SentenceTransformers)

## Results

| Classifier          | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 84.24%   | 0.84      | 0.84   | 0.84     |
| Linear SVM          | 84.54%   | 0.85      | 0.85   | 0.85     |
| Random Forest       | 78.99%   | 0.79      | 0.79   | 0.79     |

The **Linear SVM** achieved the best performance at **84.5% accuracy**, 
outperforming Logistic Regression slightly and Random Forest by a significant 
margin. Linear SVM works well with high-dimensional embedding vectors as it 
finds an optimal hyperplane to separate the classes.

## Links
- **GitHub Repo:** https://github.com/medhuuuu/ir-assignment1-embeddings-system
- **Hugging Face Dataset:** https://huggingface.co/datasets/medhuuu/movie-sentiment-dataset
- **Demo:** https://huggingface.co/spaces/medhuuu/movie-sentiment-demo

## Reflection on Working with AI
AI tools (Claude) were used extensively throughout this assignment for 
debugging environment issues, generating boilerplate code, and explaining 
concepts. The most valuable aspect was getting instant explanations of what 
embeddings actually do — converting text into vectors where similar meaning 
means similar position in space. However, AI tools also required careful 
verification; several suggested commands failed due to my Windows/PowerShell 
environment and required manual troubleshooting. Understanding each step 
before running it was essential, as the AI occasionally gave commands that 
needed adapting to my specific setup.