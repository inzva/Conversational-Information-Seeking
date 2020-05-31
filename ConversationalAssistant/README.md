# ConversationalAssistant
QueryAnswerer

This tool is created by using the MARCO dataset. 
Aim of the tool is given the question query, retrieving the most relevant answer to this question.
Our tools use 3 main processes to answer the given question
1) Finding candidate texts from the MARCO dataset, we did this step by utilizing Indri Search Interface, which retrieves relevant 
text by looking at the number of common words between the question and the queries in the MARCO dataset.
2) Re-ranking candidates texts that we found from step 1. We did this step by utilizing a similarity function tool in SpaCy.
SpaCy's similarity works by calculating word vectors and then taking their cosine similarities.
3) After retrieving the best candidates from step 2, we put them into Bert and we tried to retrieve the most relevant answer.

Requirements: MARCO Dataset and Englishstopwords.txt (any txt is okay.)

Usage: We created a web app so basically after downloading the project first you should meet the requirements and run the project
and it will work. 
