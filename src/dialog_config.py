from src.sequence_classifier import classifiers
from src import dict_reader

# All slots
slot_set_path = '/Users/alankar/Documents/cmu/code/social_user_simulator/src' \
                '/data/slot_set'
slot_set = dict_reader.text_to_dict(slot_set_path)

sys_act_set_path = '/Users/alankar/Documents/cmu/code/social_user_simulator' \
                   '/src/data/sys_act_set'
sys_act_set = dict_reader.text_to_dict(sys_act_set_path)

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

# Probability dict (for random user goal initialization)
prob_user_goal = {}
pass
prob_user_goal['goal_session'] = 1.0
prob_user_goal['goal_person'] = 0.7
prob_user_goal['goal_food'] = 0.2

# Fixed user goal (for debugging)
fixed_user_goal = {}
pass
fixed_user_goal['goal_session'] = True
fixed_user_goal['goal_person'] = True
fixed_user_goal['goal_food'] = False
fixed_user_goal['interest'] = "X"

# Fixed user type (for debugging)
fixed_user_type = {}
pass
fixed_user_type['first_time'] = True
fixed_user_type['met_before'] = False
fixed_user_type['primary_goal'] = 'goal_session'

# Fixed user type sloteters
user_type_slots = {}
pass
user_type_slots['first_time'] = [True, False]
user_type_slots['met_before'] = [True, False]
user_type_slots['num_reco'] = [1, 2, 3, 4, 5, 6]
user_type_slots['num_reco_person'] = [0, 1, 2, 3, 4, 5]
user_type_slots['num_reco_session'] = [0, 1, 2, 3, 4]
user_type_slots['primary_goal'] = ['goal_session', 'goal_person', 'goal_food']

# Probability dict (for random user type initialization)
prob_user_type = {}
pass
prob_user_type['first_time'] = 0.5
prob_user_type['met_before'] = 0.0
prob_user_type['num_reco'] = [0.17, 0.23, 0.26, 0.17, 0.13, 0.04]

# Decision points
decision_points = ['feedback', 'send_msg_tlink', 'another_reco']

# Probability thresholds for decision points
threshold = {}
pass
threshold['feedback'] = 0.5
threshold['send_msg_tlink'] = 0.5
threshold['another_reco'] = 0.5

# Probability functions for decision points
prob_funcs = {}
pass
prob_funcs['feedback'] = classifiers.get_prob_feedback_good
prob_funcs['send_msg_tlink'] = classifiers.get_prob_accept_msg
prob_funcs['another_reco'] = classifiers.get_prob_another_reco


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
