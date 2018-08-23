import random
import numpy as np
from model import CategoricalEncoder, JointEstimator
import model_params
import torch
import dialog_config
import common_func

num_reco_str = 'num_reco'
primary_goal_str = 'primary_goal'
time_str = 'time'
rapport_care_str = 'rapport_care'
cs_type_str = 'cs_type'
num_str = 'num'
goal_str = 'goal'
intention_str = 'intention'
penalty_str = 'penalty'


class RuleBasedUserSimulator:
    def __init__(self):
        self.goal = None
        self.agenda = None
        self.type = None
        self.int_state = None

        self.prev_cs_agent = None
        self.prev_cs_user = None
        self.prev_rapp_val = None
        self.prev_ti_agent = None

        self.sr = None
        self.enc = None
        self.thresh = None

    def initialize(self):
        self.type = self.generate_random_user_type()
        self.goal = self.generate_random_user_goal()
        self.agenda = self.generate_user_agenda()

        self.int_state = {}
        pass
        self.int_state['phase'] = None

        for c_slot in dialog_config.count_slots:
            self.int_state[c_slot] = 0

        cs_types = model_params.cs_types
        intention_types = model_params.intention_types

        self.enc = {}
        for p in model_params.all_participants:
            self.enc[p] = CategoricalEncoder(cs_types[p])
        self.enc[model_params.all_participants[0] + '_' + intention_str] = \
            CategoricalEncoder(intention_types[model_params.all_participants[0]])

        if dialog_config.all_together:
            index = 2
        else:
            index = 1 if self.type[rapport_care_str] else 0

        sr_type = model_params.all_participants[1]

        # Social Reasoner
        model_type = model_params.all_model_types[0]
        self.sr = self.get_model(sr_type, model_type, index=index)

        # Previous initializations
        null_list = ["NULL"]
        self.prev_cs_agent = [null_list, null_list]
        self.prev_cs_user = [null_list, null_list]
        self.prev_ti_agent = [null_list, null_list]
        self.prev_rapp_val = [0, 0]

    def get_model(self, sr_type, model_type, index):
        feature_type = model_params.feature_type[model_type][index]
        window_type = model_params.window_type[model_type][index]
        input_size = model_params.get_input_size(feature_type, window_type)
        hidden_dim = model_params.hidden_dim[model_type][index]
        leaky_slope = model_params.leaky_slope[model_type][index]
        output_size = model_params.get_output_size(model_type, sr_type)
        model_fname = model_params.model_fname[model_type][index]
        self.thresh = model_params.thresh[model_type][index]

        model = JointEstimator(input_size, hidden_dim, output_size, leaky_slope, window_type,
                               feature_type, model_type)
        model.load_state_dict(torch.load(dialog_config.data_path + model_fname, map_location='cpu'))
        return model

    def generate_random_user_goal(self):
        user_goal_slots = dialog_config.user_goal_slots
        user_goal = {}
        for slot in user_goal_slots:
            if len(user_goal_slots[slot]) == 2:
                user_goal[slot] = False
            if len(user_goal_slots[slot]) == 1:
                user_goal.update({slot: user_goal_slots[slot]})

        for slot in dialog_config.count_slots[:-1]:
            if self.type[num_str + '_' + slot] > 0:
                user_goal[goal_str + '_' + slot] = True

        return user_goal

    @staticmethod
    def generate_random_user_type():
        user_type_slots = dialog_config.user_type_slots
        prob_user_type = dialog_config.prob_user_type
        small_penalty = dialog_config.small_penalty
        large_penalty = dialog_config.large_penalty

        user_type = {}
        binary_slots = []

        for slot in user_type_slots:
            if len(user_type_slots[slot]) == 2:
                binary_slots.append(slot)

        # Pick a slot value according to the result of the biased coin toss
        for b_slot in binary_slots:
            r = random.random()
            if r < prob_user_type[b_slot]:
                user_type[b_slot] = True
            else:
                user_type[b_slot] = False

        num_recos_index = 0
        if user_type[rapport_care_str]:
            num_recos_index = 1

        if dialog_config.all_together:
            num_recos_index = 2

        # Select the total number of recos using a multinomial distribution
        x = np.random.multinomial(1, prob_user_type[num_reco_str][num_recos_index])
        num_reco = int(np.where(x == 1)[0][0] + 1)
        user_type[num_reco_str] = num_reco

        # Select the number of person recos using a multinomial distribution
        x = np.random.multinomial(1, prob_user_type[num_reco_str + "_" + str(num_reco)][num_recos_index])
        num_person = int(np.where(x == 1)[0][0])
        num_session = num_reco - num_person

        user_type[num_str + '_' + dialog_config.count_slots[1]] = num_person
        user_type[num_str + '_' + dialog_config.count_slots[0]] = num_session

        # Primary Goal
        if num_person >= num_session:
            user_type[primary_goal_str] = goal_str + '_' + dialog_config.count_slots[1]
        else:
            user_type[primary_goal_str] = goal_str + '_' + dialog_config.count_slots[0]

        # Time at hand (Less / Enough / More)
        if num_reco <= 2:
            user_type[time_str] = user_type_slots[time_str][0]
        elif num_reco <= 4:
            user_type[time_str] = user_type_slots[time_str][1]
        else:
            user_type[time_str] = user_type_slots[time_str][2]

        # If the user cares about building rapport, it will apply a small per-
        # turn penalty only if it has less time. On the other hand, if the user
        # doesn't care about building rapport, it will apply a large per-turn
        # penalty if it has less time, small per-turn penalty if it has enough
        # time and no penalty if it has more time.

        if user_type[rapport_care_str]:
            if user_type[time_str] == user_type_slots[time_str][0]:
                user_type[penalty_str] = small_penalty
            else:
                user_type[penalty_str] = 0
        else:
            if user_type[time_str] == user_type_slots[time_str][0]:
                user_type[penalty_str] = large_penalty
            elif user_type[time_str] == user_type_slots[time_str][1]:
                user_type[penalty_str] = small_penalty
            else:
                user_type[penalty_str] = 0

        return user_type
    
    @staticmethod
    def generate_user_agenda():
        return []

    def next(self, agent_action, rapport):
        r_t = self.update_agenda(agent_action, rapport)
        return self.agenda.pop(), r_t

    def update_agenda(self, agent_action, rapport):
        self.prev_rapp_val.pop()
        self.prev_rapp_val.insert(0, rapport)

        user_action = {}
        pass
        user_action['act'] = 'null'
        user_action['inform_slots'] = {}
        user_action['request_slots'] = ''

        agent_intention = common_func.get_agent_intention(agent_action)

        act = agent_action['act']
        cs = agent_action['CS'][1]

        r_t = 0

        user_action['CS'] = self.generate_conv_strat_data_driven(cs, agent_intention)

        if act == 'inform':
            inform_slots = agent_action['inform_slots'].keys()
            for c_slot in dialog_config.count_slots:
                if c_slot in inform_slots:
                    self.int_state[c_slot] += 1

        if act == 'greeting':
            user_action['act'] = 'greeting'

        elif act == 'bye':
            user_action['act'] = 'bye'

        elif act == 'request':
            user_action['act'] = 'inform'
            user_action['inform_slots'] = self.process_request_act(agent_action)

        self.agenda.append(user_action)
        r_t += self.reward_t(user_action)
        return r_t

    def generate_conv_strat_data_driven(self, agent_cs, agent_ti):
        self.prev_cs_agent.pop()
        if type(agent_cs) != list:
            agent_cs = [agent_cs]
        agent_cs = [acs.upper() for acs in agent_cs]
        self.prev_cs_agent.insert(0, agent_cs)

        self.prev_ti_agent.pop()
        if type(agent_ti) != list:
            agent_ti = [agent_ti]
        agent_ti = [ati.lower() for ati in agent_ti]
        self.prev_ti_agent.insert(0, agent_ti)

        R = torch.Tensor(np.array(self.prev_rapp_val))[None, :]
        U = torch.Tensor(np.array([self.enc['user'].fit(u) for u in self.prev_cs_user])).transpose(0, 1)[None, :, :]
        A = torch.Tensor(np.array([self.enc['agent'].fit(a) for a in self.prev_cs_agent])).transpose(0, 1)[None, :, :]
        AT = torch.Tensor(np.array([self.enc['agent_intention'].fit(at) for at in self.prev_ti_agent])).transpose(0, 1)[None, :, :]

        prob_pred = self.sr(U, A, R, AT)
        y_pred, user_cs = common_func.get_cs(self.enc['user'], self.thresh, prob_pred.data.cpu().numpy())

        self.prev_cs_user.pop()
        self.prev_cs_user.insert(0, user_cs)

        return y_pred, user_cs

    def get_feedback(self):
        rapp = self.prev_rapp_val[0]

        if rapp < 3:
            cat = 1
        elif rapp < 4:
            cat = 2
        elif rapp < 4.5:
            cat = 3
        elif rapp < 5:
            cat = 4
        else:
            cat = 5

        num_recos_index = 0
        if self.type[rapport_care_str]:
            num_recos_index = 1

        if dialog_config.all_together:
            num_recos_index = 2

        p = dialog_config.prob_user_type['acceptance_' + str(cat)][num_recos_index]
        r = random.random()
        if r < p:
            return rapp, True
        else:
            return rapp, False

    def process_request_act(self, agent_action):
        slot = agent_action['request_slots']
        decision_points = dialog_config.decision_points
        prob_funcs = dialog_config.prob_funcs

        self.int_state['phase'] = agent_action['phase']

        if slot == 'selfie':
            return {slot: True}

        for t in self.type.keys():
            if slot == t:
                return {slot: self.type[t]}

        for g in self.goal.keys():
            if slot == g:
                return {slot: self.goal[g]}

        for dp in decision_points:
            if slot == dp:
                if slot == 'send_msg_tlink':
                    rapp, val = self.get_feedback()

                elif slot == 'feedback':
                    val = True

                else:
                    val = prob_funcs[slot](self.type, self.int_state)

                return {slot: val}

    def reward_t(self, user_action):
        user_act = user_action['act']
        inform_slots = list(user_action['inform_slots'].keys())
        reward = dialog_config.reward
        reward_slots = list(reward.keys())

        # Per-turn penalty
        r_t = self.type['penalty']

        if user_act == 'inform':
            for r_slot in reward_slots:
                if r_slot in inform_slots:
                    if user_action['inform_slots'][r_slot]:
                        r_t += reward[r_slot]

        return r_t
