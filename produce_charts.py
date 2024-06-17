"""
# =============================================================================
# Sample of charts from the trainees survey
# Author: A. Vismara (DG-MF/SRF)
# Created on: 2024-05-20
# Last update on: ...
# =============================================================================

The script produces some sample charts for the trainees survey
"""

from split_groups import chart_question_by_group
from utils import q_type
from tqdm import tqdm

######
# let's say we're interested in knowing the level of reported
# contractual stress by people at different moments of their traineeship
######

# To do so, we want to split the sample along the lines of question 43:
# "For how long have you been working for the ECB?". we therefore set
question_to_split_by_nr = 43

# we are then interested in plotting the results for question 41:
# "How beneficial or harmful have your employment perspectives been to your mental health in the past month?". We set
question_to_plot_nr = 41

# for questions of privacy, we want to plot the answers for any subgroup only if there are more than 5 people in the
# subgroup. to do so when calling the plotting helper function, we set the threshold parameter to true. Setting the parameter to
# False will plot all groups regardless of the number of members

#we then call the helper function, which should open a figure in your browser
chart_question_by_group(question_to_plot_nr = question_to_plot_nr,
                        question_to_split_by_nr = question_to_split_by_nr,
                        threshold_nr_responses_per_cat = True)

# let's say now we want to plot all the questions splitting along gender lines
# let's loop over the number of questions
for i in tqdm(range(1, 51, 1)):
    if i != 44: #can't split by a question and plot the same question
        chart_question_by_group(question_to_plot_nr = i, # plot question number i from 1 to 50
                                question_to_split_by_nr = 44, # split along gender lines
                                threshold_nr_responses_per_cat = True)

# plot all the bar charts divided by gender
from random import randint
for q in tqdm(range(100)):
    q1, q2 = randint(1, 51), randint(1, 51)
    if q1 != q2: #can't split by a question and plot the same question
        chart_question_by_group(question_to_plot_nr = q1, # plot question number i from 1 to 50
                                question_to_split_by_nr = q2, # split along gender lines
                                threshold_nr_responses_per_cat = False)