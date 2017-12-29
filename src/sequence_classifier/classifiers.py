import random
from src import dialog_config


def get_prob_feedback_good(user_type, state):
    phase = state['phase']
    rapport_care = user_type['rapport_care']
    rapport_built = state['rapport_built']
    prob_feedback = dialog_config.prob_feedback
    reco_type = phase.split('_recommendation')[0]
    history = state['prev_feedback']

    feedback_good = False

    if rapport_built and rapport_care:
        index = 0
    elif not rapport_built and rapport_care:
        index = 1
    elif rapport_built and not rapport_care:
        index = 2
    else:
        index = 3

    if history is None:
        # First feedback!
        p = prob_feedback[index]['R1']
    else:
        n = state[reco_type]
        if history:
            ind = 0
        else:
            ind = 1
        p = prob_feedback[index]['R' + str(n)][ind]

    if p > random.random():
        feedback_good = True

    return feedback_good


def get_prob_accept_msg(user_type, state):
    phase = state['phase']
    rapport_care = user_type['rapport_care']
    rapport_built = state['rapport_built']
    prob_user_type = dialog_config.prob_user_type

    if rapport_built and rapport_care:
        index = 0
    elif not rapport_built and rapport_care:
        index = 1
    elif rapport_built and not rapport_care:
        index = 2
    else:
        index = 3

    accept_msg = False
    reco_type = phase.split('_recommendation')[0]

    p = prob_user_type['accept_tlink_' + reco_type][index]

    if p > random.random():
        accept_msg = True

    return accept_msg


def get_prob_another_reco(user_type, state):
    phase = state['phase']
    reco_type = phase.split('_recommendation')[0]
    another_reco = False

    if state[reco_type] < user_type['num_' + reco_type]:
        another_reco = True

    return another_reco
