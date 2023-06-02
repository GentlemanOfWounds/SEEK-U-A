# Seek-U-A
**Assessment of Quality of Chat Support Logs Using Text Mining, Sentiment Analysis, &amp; Natural Language Processing Algo's**



**Deep Learning for Natural Language Processing Frameworks Project                                                        (FEB2023 – PRESENT)**

**Title:** Assessment of Quality of Chat Support Logs Using Text Mining, Sentiment Analysis, & Natural Language Processing Algo's

**Potential Clients:** any company that offers, or is considering offering, online customer support chat services

**Data Source, Format, & Size:** publicly available 0.504 MB csv file (shape = 2_811_774 * 7) containing raw text

**CRISP-DM Project Description (inc. Business Understanding):** ***Problem:*** The global live chat market size is projected to reach $839.2 million by 2026. Businesses ranging from retailers such as Wal-Mart to online gambling corporations such as Wynn Sports to learning platforms such as projectpro.io (and beyond) use live, online employee-to-(potential) client chat conversations as essential facilitators of lucrative interactions. Although live chats boost conversions by 20%, poor interaction with (potential) customers could lead to lost sales revenue, poor client impressions, and (with the popularity and prevalence of online product review platforms) disastrous public opinion.

   ***Proposed Solution:*** An A.I.-based method of assessing the degree to which the chats increase customer satisfaction or revenue (by increasing sales or decreasing consumer churn) would be a cost-effective, objective means for a corporation to increase / maximize its profitability. In this project, we used the 0.5 MB ***“Customer Support on Twitter” Dataset*** to construct a multi-label, bidirectional long-&-short-term memory (***LSTM***)-recurrent neural network (***R.N.N.***)-based natural language processing (***N.L.P.***) model or a Bidirectional Encoder Representations from Transformers (***B.E.R.T.***) as the core of an application under design.

**CRISP-DM Business Objectives:** 0. maximize client interaction with website, thusly maximizing ad revenue; 1. maximize generalizability / utility of model; 2. minimize project length; 3. constrain project costs

**CRISP-DM Business Constraints:** 0. create high-sensitivity, high-specificity multi-label model for determination of (potential) consumer satisfaction w/ live online chat; 1. create web app for deploying such a model to analyze chat conversations uploaded by a client (in a .csv file)

**Data Science Tools Used:** Python

**Frameworks / Libraries / Packages & Modules Used:** import numpy, pandas, sklearn.feature_extraction.text's TfidfVectorizer, sklearn.feature_extraction.text's CountVectorizer, matplotlib's pyplot, wordcloud's WordCloud, wordcloud’s STOPWORDS, collections' Counter, imblearn.over_sampling's SMOTE, imblearn.under_sampling's RandomUnderSampler, sklearn.model_selection's train_test_split, re, emot.emo_unicode’s UNICODE_EMOJI, emot.emo_unicode’s EMOTICONS_EMO,
autocorrect’s Speller, nltk, nltk.corpus’ stopwords, nltk.corpus’ opinion_lexicon, nltk.stem’s WordNetLemmatizer, nltk.stem.snowball’s SnowballStemmer, nltk.tokenize’s word_tokenize, nltk.tokenize’s sent_tokenize, spacy, spacy.lang.en’s English, vaderSentiment.vaderSentiment’s SentimentIntensityAnalyzer, sklearn.ensemble’s RandomForestClassifier, sklearn.metrics’ confusion_matrix, sklearn.metrics’ accuracy_score, sklearn.metrics’ classification_report, tensorflow, tensorflow.keras.models’ Sequential, tensorflow.keras’ layers, tensorflow.keras’ Input, tensorflow.keras.preprocessing.sequence’s pad_sequences, tensorflow.keras.preprocessing.text’s Tokenizer, tensorflow.keras.utils’ to_categorical, tensorflow.keras.preprocessing’s text_dataset_from_directory, tensorflow.keras.layers.experimental.preprocessing’s TextVectorization, tensorflow.keras.layers’ Embedding, tensorflow.keras.layers’  LSTM, pickle, flask

**Steps Taken:** extensive data cleansing / natural language preprocessing (inc. removing stop-words, stemming / lemmatization, tokenization, & replacing emojis w/ words to capture their sentiment), exploratory data analysis via wordclouds, sentiment adjudication & quantification, group balancing, text vectorization, random forest-based classification, machine learning model assessment, ***Bidirectional LSTM-R.N.N.***-based classification, ***B.E.R.T.***-based classification, deployment

**Roles and Responsibilities:** jointly responsible for conceptualization, research, design, end-to-end implementation (inc. model building) and evaluation

**CRISP-DM Business Benefits:** 
0. Provided managerial clients w/ ***simple, high-quality, cost-free means of assessing employee performance in online chats***
1. Afforded employees w/ ***objective, third-party means of performance evaluation***; which they could use for self-evaluation to improve chat performance and/or to prepare for professional evaluation
2. Created **revenue-generating website*** for self & other stakeholders
