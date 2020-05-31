import operator
from concurrent import futures
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import requests
from bs4 import BeautifulSoup
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap
from flask import Flask, request, render_template, redirect
import os
import numpy as np
import pandas as pd
import ast
import spacy
from tqdm import tqdm
from functools import lru_cache
import json

import spacy.cli

#spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")

path = "/Users/cerendardagan/Desktop/inzva/ConversationalAssistant-master/collection.tsv"  # MARCO
df = pd.read_table(path, header=None, index_col=0)

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
df_scores = pd.read_csv("AnswersAndScores.txt")

# Preprocessing the Query

with open('/Users/cerendardagan/Desktop/inzva/ConversationalAssistant-master/englishStopwords.txt',
          'r') as f:  # herhangi bir stopwords iceren text sufficient
    myLists = [line.strip() for line in f]

app = Flask(__name__)
Bootstrap(app)
app.config['SECRET_KEY'] = "DontTellAnyone"


class SearchForm(FlaskForm):
    name = StringField('query', validators=[DataRequired()])


@app.route('/', methods=('GET', 'POST'))
def my_form():
    form = SearchForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data  # input alınan query
        answer_gen, last_can_gen, ndcg_score_gen, indexes_gen,similarity_results_gen, answer_scoring, last_can_scoring, ndcg_score_scoring,similarity_results_scoring= change(name)  #
        similarity_results_gen = similarity_results_gen[::-1]
        similarity_results_scoring = similarity_results_scoring[::-1]

        return render_template('new-page.html', form=form, answer_gen=answer_gen, score_gen=ndcg_score_gen,
                               last_can_gen=last_can_gen,indexes_gen=indexes_gen,similarity_results_gen=similarity_results_gen,answer_scoring=answer_scoring, score_scoring=ndcg_score_scoring,similarity_results_scoring=similarity_results_scoring,
                               last_can_scoring=last_can_scoring)  # answer,score,query sayfasını dönüyor
    else:
        return render_template('submit.html', form=form)  # query girilmezse aynı sayfada kalıyor

@lru_cache(maxsize=1024)
def change(query):  # sonuç burdan dönüyor
    preprocessed = preprocess(query)
    data_gen = request_MARCO(preprocessed)
    last_can_gen,similarity_results_gen,indexes_gen  = similarity(preprocessed, data_gen)
    answer_gen = answer_question(query, last_can_gen)
    score_gen,ndcg_score_gen,scores_check_gen = output_score(similarity_results_gen,answer_gen,last_can_gen,data_gen,preprocessed,indexes_gen)
    sim_graph_gen =[list(pair) for pair in zip(indexes_gen, similarity_results_gen)]
    sim_graph_gen.insert(0,['Index','Similarity Score'])

    data_scoring = df_scores['Passage_Content']
    last_can_scoring,similarity_results_scoring,indexes_scoring  = similarity(preprocessed, data_scoring)
    answer_scoring = answer_question(query, last_can_scoring)

    score_scoring,ndcg_score_scoring,scores_check_scoring = output_score(similarity_results_scoring,answer_scoring,last_can_scoring,data_scoring,preprocessed,indexes_scoring)

    return answer_gen, last_can_gen, ndcg_score_gen,indexes_gen,similarity_results_gen,answer_scoring, last_can_scoring, ndcg_score_scoring,similarity_results_scoring




def output_score(similarity_results,answer,last_can,data,preprocessed,indexes):  # burayı eklersiniz
    scores = []
    scores_check = []
    for i in indexes:
        select_indices = list(df_scores["Passage_Content"] == data[i])

        if True in select_indices :
            scores.append(max(df_scores.iloc[select_indices]["Score"]))
            scores_check.append(1)
        else:
            scores.append(0)
            scores_check.append(-1)
    ndcg_score = ndcg_at_k(scores,len(scores))
    
    return scores,ndcg_score,scores_check

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]),  # The tokens representing our input text.
                                     token_type_ids=torch.tensor(
                                         [segment_ids]))  # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    return answer


def preprocess(query):
    cand_list = []
    list_query = query.split(' ')
    for element in list_query:
        if not element in myLists:
            cand_list.append(element)
    return ' '.join(cand_list)


def request_MARCO(preprocessed):
    list_of_urls = []

    for i in range(40):
        list_of_urls.append('http://boston.lti.cs.cmu.edu/Services/treccast19/lemur.cgi?d=0&s=+' + str(
            i * 10) + '&n=10&q=' + preprocessed)

    url = 'http://boston.lti.cs.cmu.edu/Services/treccast19/lemur.cgi?d=0&s=0&n=400&q=' + preprocessed

    from concurrent import futures
    with futures.ThreadPoolExecutor(
            max_workers=40) as executor:  ## you can increase the amount of workers, it would increase the amount of thread created
        res = executor.map(requests.get, list_of_urls)
    responses = list(res)  ## the

    solmenu = []
    for i in responses:
        source = BeautifulSoup(i.content)
        solmenu.append((source.find_all('font')))

    """# Intersectionining the ids that comes from the request and all of the MARCO"""

    ids = []
    for elem in solmenu:
        for item in elem:
            if str(item).find('MARCO') != -1:
                process = str(item)
                aydi = process[process.find('MARCO') + 6:process.find(')')]
                if aydi[0] != ' ':
                    ids.append(aydi)
                # print(process[process.find('MARCO'):process.find('>')-1])
                # print('******')

    df.loc[int(ids[1]), 1]
    ids = ids[1:]
    data = []
    for aydi in ids:
        ayd = int(aydi)
        elem = df.loc[ayd, 1]
        data.append(str(elem))
    return data

def similarity(preprocessed, data):
    doc = nlp(preprocessed)
    sim = []
    for d in tqdm(data):
        doc1 = nlp(d)
        sim.append(doc1.similarity(doc))

        # candidates[elem] = (sim)

    import operator
    index, value = max(enumerate(sim), key=operator.itemgetter(1))

    indexes = sorted(range(len(sim)), key=lambda i: sim[i])[-10:]
    # burasi degisecek birazcik 512ye kadarini al tarzi bir sey yapmam daha uygun olur gibi
    last_can = ''
    str_cnt = ''
    similarity_results  =list(sorted(sim)[-10:])

    for i in indexes:

        str_cnt += data[i] + ' '
        input_ids = tokenizer.encode('bu bir denemedir', str_cnt)
        if len(input_ids) < 512:
            last_can += data[i] + ' '
        else:
            return last_can, similarity_results, indexes

    return last_can, similarity_results, indexes

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg


if __name__ == '__main__':
    app.run(debug=True)
