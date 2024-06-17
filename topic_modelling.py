import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import re
import os
from pprint import pprint

from collections import Counter
import matplotlib.colors as mcolors

# Gensim
import gensim, logging, warnings
import gensim.corpora as corpora
from gensim.utils import  simple_preprocess
from gensim.models import CoherenceModel

warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

###############

## change this to the directory where you saved the data
file_loc = "c://users//avism//desktop//trainees_survey//data_2405//survey_results.xlsx"

#load the data in as a dataframe
survey_data = pd.read_excel(file_loc, usecols = "J:DK")

headers = survey_data.loc[0]
survey_data = survey_data.drop(0)

###topic modelling on priorities
#import words/punctuation to clean
punctuation = """!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@#"""
stopwords = open(f"{os.getcwd()}//stopwords.txt", "r").read().split("\n")
numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

#function to cleaning answers for one question
def clean_answer(answer):
    answer = answer.lower()
    answer = re.sub('accomodation', 'accommodation', answer)
    answer = re.sub('['+punctuation + ']+', '', answer) #strip punctuation
    answer = [word for word in answer.split(' ')
     if word not in stopwords]
    answer = [word for word in answer
     if word not in numbers]
    answer = ' '.join(answer)
    return answer

def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)

def format_topics_sentences(lda_model, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(lda_model[corpus]):
        row = row_list[0] if lda_model.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df._append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def topic_modelling_answer(answer, number_of_topics = 4):
    """applies topic modelling to a dataframe series.
    Can be a single answer or a collection of answers. In the
    latter case, these need to first be merged into a single series"""
    series = answer.dropna().apply(lambda x: [clean_answer(x)]).to_list()

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(series, min_count=5, threshold=10)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[series], threshold=10)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    texts = [[word for word in simple_preprocess(str(doc))] for doc in series]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]

    # Create Dictionary
    id2word = corpora.Dictionary(texts)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=6,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=8,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)

    pprint(lda_model.print_topics())

    df_topic_sents_keywords = format_topics_sentences(lda_model=lda_model, corpus=corpus, texts=texts)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic.head(10)

    # Display setting to show more characters in column
    pd.options.display.max_colwidth = 100

    sent_topics_sorteddf_mallet = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
                                                axis=0)

    # Reset Index
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

    # Show
    sent_topics_sorteddf_mallet.head(10)

    doc_lens = [len(d) for d in df_dominant_topic.Text]

    # Plot
    plt.figure(figsize=(9, 6), dpi=160)
    plt.hist(doc_lens, bins=30, color='navy')
    plt.text(100, 5, "Mean   : " + str(round(np.mean(doc_lens))))
    plt.text(100, 4.5, "Median : " + str(round(np.median(doc_lens))))
    plt.text(100, 4, "Stdev   : " + str(round(np.std(doc_lens))))
    plt.text(100, 3.5, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
    plt.text(100, 3, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

    plt.gca().set(xlim=(0, 200), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0, 200, 9))
    plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
    plt.show()

    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in texts for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(3, 2, figsize=(8, 5), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
               label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                    label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.030);
        ax.set_ylim(0, 30)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=10)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
        ax.legend(loc='upper left');
        ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=12, y=1.05)
    plt.show()

    return df_topic_sents_keywords, df_dominant_topic, df

### here you select the answer or asnwers you want to do the topic modelling for
series = survey_data.iloc[:, 4] #col 4 has the open answer for priorities

try:
    df_series = pd.DataFrame([])
    for column in series.columns:
        df_series = pd.concat([df_series, series[column]])
except:
    df_series = pd.DataFrame([])

df_topic_sents_keywords, df_dominant_topic, df_topics = topic_modelling_answer(df_series.iloc[:, 0] if series.size > 300 else series)


##worldcloud
if series.size > 300:
    series = df_series.iloc[:, 0]
else:
    series = series
text = " ".join(i for i in series.dropna().apply(lambda x: clean_answer(x)))
wordcloud = WordCloud(background_color="white").generate(text)
wordcloud.to_file('words.png')
plt.figure(figsize = (15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
