from src.sequence_classifier import classifiers
from src import dict_reader

data_path = "/Users/alankar/Documents/cmu/code/social_user_simulator/src/data/"
# All slots
slot_set_path = data_path + "slot_set"
slot_set = dict_reader.text_to_dict(slot_set_path)

sys_act_set_path = data_path + "sys_act_set"
sys_act_set = dict_reader.text_to_dict(sys_act_set_path)

true_reward_path = data_path + "true_reward.csv"
true_turn_path = data_path + "true_turn.csv"

# Dialog rewards
reward = {}
pass
reward['feedback'] = 5
reward['send_msg_tlink'] = 10

# Count tracking
count_slots = ['session', 'person', 'food']

# User Goal slots
user_goal_slots = {}
pass
user_goal_slots['interest'] = "X"
user_goal_slots['goal_food'] = [True, False]
user_goal_slots['goal_person'] = [True, False]
user_goal_slots['goal_session'] = [True, False]

# User Type slots
user_type_slots = {}
pass
user_type_slots['first_time'] = [True, False]
user_type_slots['met_before'] = [True, False]
user_type_slots['num_reco'] = [1, 2, 3, 4, 5, 6]
user_type_slots['num_reco_person'] = [0, 1, 2, 3, 4, 5, 6]
user_type_slots['num_reco_session'] = [0, 1, 2, 3, 4, 5, 6]
user_type_slots['primary_goal'] = ['goal_session', 'goal_person', 'goal_food']
# Less Time (LT): 1-2 recos, Enough Time (ET): 3-4 recos, More Time (MT): 5-6
# recos
user_type_slots['time'] = ['LT', 'ET', 'MT']
# Whether or not the user cares about building rapport
user_type_slots['rapport_care'] = [True, False]
# Type-1: None, Type-2: SD-QESD Type, Type-3: SD-PR Type, Type-4: PR-HE Type
user_type_slots['cs_type'] = [1, 2, 3, 4]

# Probability dict (for random user type initialization)
prob_user_type = {}
pass
prob_user_type['first_time'] = 0.5
prob_user_type['met_before'] = 0.0
# Probability that user cares about building rapport indexed by the number of
# recommendations the user is going to ask for (1 to 6).
prob_user_type['rapport_care'] = [0.25, 0.5, 0.5, 0.58, 0.44, 0.5]
# prob_user_type['rapport_care'] = [1, 1, 1, 1, 1, 1]
# Probability distribution for total number of recos (1 to 6)
prob_user_type['num_reco'] = [0.17, 0.23, 0.26, 0.17, 0.13, 0.04]
# Probability distribution for total number of person recos (0 to num_reco)
prob_user_type['num_reco_1'] = [0.5, 0.5]
prob_user_type['num_reco_2'] = [0.125, 0.1875, 0.6875]
prob_user_type['num_reco_3'] = [0.17, 0.33, 0.28, 0.22]
prob_user_type['num_reco_4'] = [0.08, 0.08, 0.5, 0.25, 0.09]
prob_user_type['num_reco_5'] = [0, 0, 0.11, 0.67, 0.22, 0]
prob_user_type['num_reco_6'] = [0, 0, 0, 0, 1, 0, 0]
# Probability distribution for user CS personality/types
prob_user_type['cs_type'] = [0.25, 0.25, 0.25, 0.25]

# Probability distribution for agent CS
prob_agent_cs = [0.2, 0.2, 0.2, 0.2, 0.2]
agent_cs = ['None', 'SD', 'QESD', 'PR', 'HE']

# Agent CS for each user type should be above these thresholds for rapport to
# be built
threshold_cs = [0.3, 0.5, 0.5, 0.5]

# Probability of acceptance for 4 cases:
# 0: Rapport built, care
# 1: Rapport not built, care
# 2: Rapport built, not care
# 3: Rapport not built, not care
prob_user_type['accept_tlink_person'] = [0.8, 0.625, 0.25, 0.74]
prob_user_type['accept_tlink_session'] = [1, 0.8, 0.67, 0.89]

prob_feedback = {
    0: {
        'R1': 0.63,
        'R2': [1, 1], # Entry-1 is P(R2|R1), Entry-2 is P(R2|~R1)
        'R3': [0.83, 0],
        'R4': [1, 0],
        'R5': [0, 0],
        'R6': [0, 0]
    },
    1: {
        'R1': 0.32,
        'R2': [0.25, 0.43],  # Entry-1 is P(R2|R1), Entry-2 is P(R2|~R1)
        'R3': [0.14, 0.5],
        'R4': [0, 0],
        'R5': [0, 0.4],
        'R6': [0, 0]
    },
    2: {
        'R1': 0.22,
        'R2': [0, 0.8],  # Entry-1 is P(R2|R1), Entry-2 is P(R2|~R1)
        'R3': [0, 1],
        'R4': [0, 0],
        'R5': [0, 0],
        'R6': [0, 0]
    },
    3: {
        'R1': 0.83,
        'R2': [0.82, 1],  # Entry-1 is P(R2|R1), Entry-2 is P(R2|~R1)
        'R3': [0.73, 1],
        'R4': [0.7, 1],
        'R5': [0.8, 1],
        'R6': [0, 0]
    }
}

# Decision points
decision_points = ['feedback', 'send_msg_tlink', 'another_reco']

# # Probability thresholds for decision points
# threshold = {}
# pass
# threshold['feedback'] = 0.5
# threshold['send_msg_tlink'] = 0.5
# threshold['another_reco'] = 0.5

# Probability functions for decision points
prob_funcs = {}
pass
prob_funcs['feedback'] = classifiers.get_prob_feedback_good
prob_funcs['send_msg_tlink'] = classifiers.get_prob_accept_msg
prob_funcs['another_reco'] = classifiers.get_prob_another_reco

# Penalty per turn
small_penalty = -0.25
large_penalty = -1

# print(prob_funcs)

# Print action
def print_info(action):
    act = action['act']

    if act == 'inform':
        act_slots = action[act + '_slots']
        print(act, end='')
        for slot in act_slots:
            print('(' + str(slot) + '=' + str(act_slots[slot]), end='')
        print(')')

    elif act == 'request':
        act_slots = action[act + '_slots']
        print(act, end='')
        print('(' + act_slots, end='')
        print(')')
    else:
        print(act + '()')


# Config parameters for RL-based agent
sys_request_slots = list(slot_set.keys())[0:10]
sys_inform_slots = list(slot_set.keys())[-3:]

sys_only_acts = list(sys_act_set.keys())[0:8]

# Feasible actions
feasible_actions = []

# Add only-act actions to the feasible_actions list
for act in sys_only_acts:
    feasible_actions.append(
        {
            'act': act,
            'request_slots': '',
            'inform_slots': {}
        }
    )

# Add request actions to the feasible_actions list
for slot in sys_request_slots:
    feasible_actions.append({'act': 'request', 'request_slots': slot})

# Add inform actions to the feasible_actions list
for slot in sys_inform_slots:
    feasible_actions.append(
        {
            'act': 'request',
            'inform_slots': {slot: 'info_' + slot}
        }
    )
