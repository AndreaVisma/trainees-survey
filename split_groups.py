"""
# =============================================================================
# Sample of charts from the trainees survey
# Author: A. Vismara (DG-MF/SRF)
# Created on: 2024-05-20
# Last update on: ...
# =============================================================================

The script creates some functions to output charts from the trainees survey questions, by splitting the sample along the categories of any question
e.g. we want to know the difference between men and women in answering any question
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import q_cols_order, q_type

colors = ['rgba(39, 120, 245, 0.8)', 'rgba(39, 208, 245, 0.8)',
          'rgba(39, 183, 245, 0.3)', 'rgba(245, 93, 39, 0.3)',
          'rgba(245, 93, 39, 0.8)', 'rgba(245, 67, 39, 1)']
colors_many = ['rgba(39, 120, 245, 0.8)', 'rgba(39, 208, 245, 0.8)',
          'rgba(39, 183, 245, 0.3)', 'rgba(245, 93, 39, 0.3)',
          'rgba(245, 93, 39, 0.8)', 'rgba(245, 67, 39, 1)',
            'rgba(139, 0, 219, 0.7)', 'rgba(206, 196, 195, 0.5)']


## change this to the directory where you saved the data
file_loc = os.getcwd() + "data\\survey_results.xlsx"

#load the data in as a dataframe
survey_data = pd.read_excel(file_loc, usecols = "J:DK")
##cleanup some stuff
survey_data.replace('Strongly agre','Strongly agree', inplace = True)
survey_data.replace('Strongly Agree','Strongly agree', inplace = True)
survey_data.replace('Strongy Agree','Strongly agree', inplace = True)
survey_data.replace('200-149 EUR','200-249 EUR', inplace = True)

headers = survey_data.loc[0].reset_index()
questions = headers[~headers['index'].str.contains('Unnamed')]
headers.loc[headers['index'].str.contains('Unnamed'), 'index'] = np.nan
headers['index'] = headers['index'].fillna(method = 'ffill')
dict_questions = dict(zip([x + 1 for x in range(len(questions))], questions['index']))

def chart_question_by_group(question_to_split_by_nr,
                            question_to_plot_nr,
                            threshold_nr_responses_per_cat = True):
    index_split = headers[headers['index'] == dict_questions[question_to_split_by_nr]].index.to_list()
    index_question = headers[headers['index'] == dict_questions[question_to_plot_nr]].index.to_list()

    if len(index_split) == 1 and (q_type[question_to_split_by_nr] != 'quali'):
        if (q_type[question_to_plot_nr] == 'likert'):
            df = survey_data.iloc[:,index_split + index_question].dropna()

            #drop when too few respondents per category
            if threshold_nr_responses_per_cat:
                df_group = df.groupby(dict_questions[question_to_split_by_nr]).count()
                ind_to_drop = df_group[df_group[dict_questions[question_to_plot_nr]] < 5].index.to_list()
            else:
                ind_to_drop = []

            ##plot by question type
            #likert scale, onlly one answer
            if (len(index_question) == 1):
                df = df.drop(0)
                df = df[~df[dict_questions[question_to_split_by_nr]].isin(ind_to_drop)]
                plot_one_answer_likert(df, question_to_split_by_nr, question_to_plot_nr)

            #likert scale more than one sub-questions
            else:
                questions = df.iloc[0, 1:].to_list()
                if question_to_plot_nr != 16:
                    questions = [x for x in questions if 'comment' not in x] #carry over because theý ll be useful later
                df = df.drop(0)
                df = df[~df[dict_questions[question_to_split_by_nr]].isin(ind_to_drop)]
                plot_many_answers_likert(df, questions, question_to_split_by_nr, question_to_plot_nr)

        # bar chart
        if (q_type[question_to_plot_nr] == 'bars'):
            ## case when there are multiple options
            if (len(index_question) > 1) and (question_to_plot_nr not in [21, 22, 34]):
                df = survey_data.iloc[:, index_split + index_question]
                cats = df.iloc[0, 1:].to_list()
                df = df.drop(0)
                df = df.set_index(dict_questions[question_to_split_by_nr])
                df.rename(columns = dict(zip(df.columns, cats)), inplace  =True)
                for x in cats: df.loc[~df[x].isna(), x] = 1
                df.fillna(0, inplace = True)
                df = df.reset_index().dropna()

                df_group = df.groupby(dict_questions[question_to_split_by_nr]).count()
                totals = dict(zip(df_group.index, df_group.iloc[:, 0]))

                # drop when too few respondents per category
                if threshold_nr_responses_per_cat:
                    ind_to_drop = df_group[df_group.iloc[:,0] < 5].index.to_list()
                else:
                    ind_to_drop = []

                df = df[~df[dict_questions[question_to_split_by_nr]].isin(ind_to_drop)]
                plot_multiple_answers_bars(df, question_to_split_by_nr, question_to_plot_nr, totals)

            ##case where there were no multiple options
            else:
                if question_to_plot_nr in [21, 22]:
                    df = survey_data.iloc[:, index_split + index_question].iloc[:,:2].dropna()
                else:
                    df = survey_data.iloc[:, index_split + index_question].dropna()

                # drop when too few respondents per category
                if threshold_nr_responses_per_cat:
                    df_group = df.groupby(dict_questions[question_to_split_by_nr]).count()
                    ind_to_drop = df_group[df_group.iloc[:,0] < 5].index.to_list()
                else:
                    ind_to_drop = []
                df = df.drop(0)
                df = df[~df[dict_questions[question_to_split_by_nr]].isin(ind_to_drop)]
                plot_one_answer_bars(df, question_to_split_by_nr, question_to_plot_nr)

    else:
        print('tried to split by a question that contains subquestions or is qualitative. not implemented yet ...')

def plot_one_answer_likert(df, question_to_split_by_nr, question_to_plot_nr):

    df_group = df.groupby([dict_questions[question_to_split_by_nr],
                           dict_questions[question_to_plot_nr]]).size().reset_index()
    df_group.rename(columns = {0:'count'}, inplace = True)
    for group in df_group[dict_questions[question_to_split_by_nr]].unique():
        tot_group = df_group.loc[df_group[dict_questions[question_to_split_by_nr]] == group,
        'count'].sum()
        df_group.loc[df_group[dict_questions[question_to_split_by_nr]] == group,
        'count'] = 100 * df_group.loc[df_group[dict_questions[question_to_split_by_nr]] == group,
        'count'] / tot_group
    df_pivot = df_group.pivot(index = dict_questions[question_to_split_by_nr],
                              columns = dict_questions[question_to_plot_nr],
                              values = 'count')
    df_pivot.fillna(0, inplace = True)

    ##order columns
    if q_cols_order[question_to_plot_nr] != False:
        if len(q_cols_order[question_to_plot_nr]) - len(df_pivot.columns) > 0:
            cols = [x for x in q_cols_order[question_to_plot_nr] if x in df_pivot.columns.tolist()]
            extra_cols = len(q_cols_order[question_to_plot_nr]) - len(df_pivot.columns)
            df_pivot = df_pivot[list(cols)]
        else:
            extra_cols = 0
            df_pivot = df_pivot[q_cols_order[question_to_plot_nr]]
    else:
        extra_cols = 0

    df_pivot = df_pivot.reset_index()

    fig = go.Figure()
    if len(q_cols_order[question_to_plot_nr]) > 6:
        colormap = colors_many if question_to_plot_nr != 1 else [f'rgba {plt.get_cmap()(x)}' for x in np.linspace(0, 1, 11)]
    else:
        colormap = colors
    for i in range(len(df_pivot.columns[1:])):
        col = df_pivot.columns[i + 1]
        if question_to_plot_nr != 18:
            color = colormap[i + extra_cols] if 'say' not in str(col) else 'rgba(206, 196, 195, 0.5)'
        else:
            color = colormap[i + extra_cols] if 'apply' not in str(col) else 'rgba(206, 196, 195, 0.5)'
        fig.add_trace(go.Bar(x=df_pivot[col].values,
                             y=df_pivot[df_pivot.columns[0]],
                             text = df_pivot[col].astype('int') / 100,
                             orientation='h',
                             name=col if type(col) == 'str' else str(col),
                             customdata=df_pivot[col].astype('int'),
                             hovertemplate="%{y}: %{customdata}",
                             marker = dict(color = color)))

    fig.update_traces(texttemplate='%{text:.0%}', textposition='inside',
                      insidetextanchor = 'middle', textfont_size=16)
    fig.update_layout(barmode='stack',
                      bargap=0.04,
                      legend_orientation='h',
                      height=800,
                      width=1700,
                      title = dict_questions[question_to_plot_nr],
                      yaxis_title = dict_questions[question_to_split_by_nr],
                      legend={'traceorder': 'normal'},
                      margin=dict(l=100, r=20, t=80, b=20)
                      )
    fig.update_xaxes(range=[0, 100], showticklabels = False, showgrid = False)

    fig.show()

def plot_many_answers_likert(df, questions, question_to_split_by_nr, question_to_plot_nr):

    fig = make_subplots(rows=1 + (len(questions) - 1) // 3, cols=3,
                        subplot_titles = questions)

    for i in range(len(questions)):
        df_group = df.iloc[:, [0, i + 1]].groupby([dict_questions[question_to_split_by_nr],
                               df.iloc[:, i + 1].name]).size().reset_index()
        df_group.rename(columns = {0:'count'}, inplace = True)
        for group in df_group[dict_questions[question_to_split_by_nr]].unique():
            tot_group = df_group.loc[df_group[dict_questions[question_to_split_by_nr]] == group,
            'count'].sum()
            df_group.loc[df_group[dict_questions[question_to_split_by_nr]] == group,
            'count'] = 100 * df_group.loc[df_group[dict_questions[question_to_split_by_nr]] == group,
            'count'] / tot_group
        df_pivot = df_group.pivot(index = dict_questions[question_to_split_by_nr],
                                  columns = df.iloc[:, i + 1].name,
                                  values = 'count')
        df_pivot.fillna(0, inplace = True)

        ##order columns
        if q_cols_order[question_to_plot_nr] != False:
            if len(q_cols_order[question_to_plot_nr]) - len(df_pivot.columns) > 0:
                cols = [x for x in q_cols_order[question_to_plot_nr] if x in df_pivot.columns.tolist()] #checks if all possible answers were given
                ind_cols = [q_cols_order[question_to_plot_nr].index(x) for x in q_cols_order[question_to_plot_nr] if x in df_pivot.columns.tolist()]
                df_pivot = df_pivot[list(cols)]
            else:
                ind_cols = [q_cols_order[question_to_plot_nr].index(x) for x in q_cols_order[question_to_plot_nr] if x in df_pivot.columns.tolist()]
                df_pivot = df_pivot[q_cols_order[question_to_plot_nr]]
        else:
            ind_cols = [q_cols_order[question_to_plot_nr].index(x) for x in q_cols_order[question_to_plot_nr] if x in df_pivot.columns.tolist()]

        df_pivot = df_pivot.reset_index()

        if len(q_cols_order[question_to_plot_nr]) > 6:
            colormap = colors_many
        else:
            colormap = colors
        for t in range(len(df_pivot.columns[1:])):
            col = df_pivot.columns[t + 1]
            color = colormap[ind_cols[t]] if 'say' not in str(col) else 'rgba(206, 196, 195, 0.5)'
            fig.add_trace(go.Bar(x=df_pivot[col].values,
                                 y=df_pivot[df_pivot.columns[0]],
                                 text = df_pivot[col].astype('int') / 100,
                                 orientation='h',
                                 name=col if type(col) == 'str' else str(col),
                                 customdata=df_pivot[col].astype('int'),
                                 hovertemplate="%{y}: %{customdata}",
                                 marker = dict(color = color),
                                 showlegend=True if i < 1 else False), row =1 + i // 3, col = (1 + i) - (3 * (i // 3)))

    fig.update_traces(texttemplate='%{text:.0%}', textposition='inside',
                      insidetextanchor = 'middle', textfont_size=16)
    fig.update_layout(barmode='stack',
                      bargap=0.04,
                      legend_orientation='h',
                      title = dict_questions[question_to_plot_nr],
                      legend={'traceorder': 'normal'},
                      margin=dict(l=150, r=20, t=80, b=20)
                      )
    fig.add_annotation(text=dict_questions[question_to_split_by_nr],
                       xref="x domain", yref="paper",
                       x=-0.25, y=0.5, showarrow=False,
                       textangle=-90,
                       font=dict(size = 16))

    fig.update_xaxes(range=[0, 100], showticklabels = False, showgrid = False)

    fig.show()

def plot_one_answer_bars(df, question_to_split_by_nr, question_to_plot_nr):

    df_group = df.groupby([dict_questions[question_to_split_by_nr],
                           dict_questions[question_to_plot_nr]]).size().reset_index()
    df_group.rename(columns = {0:'count'}, inplace = True)
    for group in df_group[dict_questions[question_to_split_by_nr]].unique():
        tot_group = df_group.loc[df_group[dict_questions[question_to_split_by_nr]] == group,
        'count'].sum()
        df_group.loc[df_group[dict_questions[question_to_split_by_nr]] == group,
        'count'] = 100 * df_group.loc[df_group[dict_questions[question_to_split_by_nr]] == group,
        'count'] / tot_group
    df_pivot = df_group.pivot(index = dict_questions[question_to_split_by_nr],
                              columns = dict_questions[question_to_plot_nr],
                              values = 'count')
    df_pivot.fillna(0, inplace = True)

    ##order columns
    if q_cols_order[question_to_plot_nr] != False:
        if len(q_cols_order[question_to_plot_nr]) - len(df_pivot.columns) > 0:
            cols = [x for x in q_cols_order[question_to_plot_nr] if x in df_pivot.columns.tolist()]
            df_pivot = df_pivot[list(cols)]
        else:
            df_pivot = df_pivot[q_cols_order[question_to_plot_nr]]

    df_pivot = df_pivot.reset_index()

    fig = go.Figure()
    if len(df_pivot) > 6:
        colormap = colors_many
    else:
        colormap = colors
    for i in range(len(df_pivot)):
        color = colormap[i]
        fig.add_trace(go.Bar(x=df_pivot.iloc[i, 1:].values,
                             y=df_pivot.columns[1:],
                             text = df_pivot.iloc[i, 1:].values.astype('int') / 100,
                             orientation='h',
                             name=df_pivot.iloc[i, 0] if type(df_pivot.iloc[i, 0]) == 'str' else str(df_pivot.iloc[i, 0]),
                             customdata=df_pivot.iloc[i, 1:].values.astype('int'),
                             hovertemplate="%{y}: %{customdata}",
                             marker = dict(color = color)))

    fig.update_traces(texttemplate='%{text:.0%}', textposition='inside',
                      insidetextanchor = 'middle', textfont_size=16)
    fig.update_layout(bargap=0.2,
                      legend_orientation='h',
                      height=800,
                      width=1700,
                      title = dict_questions[question_to_plot_nr],
                      legend={'traceorder': 'normal'},
                      margin=dict(l=100, r=20, t=80, b=20)
                      )
    fig.add_annotation(text=dict_questions[question_to_split_by_nr],
                       xref="x domain", yref="paper",
                       x=0.01, y=-0.08, showarrow=False,
                       font=dict(size = 14))
    fig.update_xaxes(range=[0, 100], showticklabels = False, showgrid = False)

    fig.show()

def plot_multiple_answers_bars(df, question_to_split_by_nr, question_to_plot_nr, totals):

    df_group = df.groupby(dict_questions[question_to_split_by_nr]).sum().reset_index()

    for group in df_group[dict_questions[question_to_split_by_nr]].unique():
        for col in df_group.columns[1:]:
            df_group.loc[df_group[dict_questions[question_to_split_by_nr]] == group,
                col] = 100 * df_group.loc[df_group[dict_questions[question_to_split_by_nr]] == group,
                col] / totals[group]

    df_group.fillna(0, inplace = True)
    df_group.set_index(dict_questions[question_to_split_by_nr], inplace = True)

    ##order columns
    if q_cols_order[question_to_plot_nr] != False:
        if len(q_cols_order[question_to_plot_nr]) - len(df_group.columns) > 0:
            cols = [x for x in q_cols_order[question_to_plot_nr] if x in df_group.columns.tolist()]
            df_group = df_group[list(cols)]
        else:
            df_group = df_group[q_cols_order[question_to_plot_nr]]

    df_group.reset_index(inplace = True)

    fig = go.Figure()
    if len(df_group) > 6:
        colormap = colors_many
    else:
        colormap = colors
    for i in range(len(df_group)):
        color = colormap[i]
        fig.add_trace(go.Bar(x=df_group.iloc[i, 1:].values,
                             y=df_group.columns[1:],
                             text = df_group.iloc[i, 1:].values.astype('int') / 100,
                             orientation='h',
                             name=df_group.iloc[i, 0] if type(df_group.iloc[i, 0]) == 'str' else str(df_group.iloc[i, 0]),
                             customdata=df_group.iloc[i, 1:].values.astype('int'),
                             hovertemplate="%{y}: %{customdata}",
                             marker = dict(color = color)))

    fig.update_traces(texttemplate='%{text:.0%}', textposition='inside',
                      insidetextanchor = 'middle', textfont_size=16)
    fig.update_layout(bargap=0.2,
                      legend_orientation='h',
                      height=800,
                      width=1700,
                      title = dict_questions[question_to_plot_nr],
                      legend={'traceorder': 'normal'},
                      margin=dict(l=100, r=20, t=80, b=20)
                      )
    fig.add_annotation(text=dict_questions[question_to_split_by_nr],
                       xref="x domain", yref="paper",
                       x=0.01, y=-0.08, showarrow=False,
                       font=dict(size = 14))
    fig.update_xaxes(range=[0, 100], showticklabels = False, showgrid = False)

    fig.show()