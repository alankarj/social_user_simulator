def get_prob_accept_msg(user_type, state):
    return True


def get_prob_another_reco(user_type, state):  # This will remain the same
    phase = state['phase']
    reco_type = phase.split('_recommendation')[0]

    another_reco = False

    if reco_type == 'food':
        return False

    if state[reco_type] < user_type['num_' + reco_type]:
        another_reco = True

    return another_reco
