from src import dialog_config
import random
import numpy as np
import json


class RuleBasedUserSimulator:
    def __init__(self):
        # goal is a dict containing inform slots and request slots
        self.goal = None
        # agenda is a stack representing pending user actions
        self.agenda = None
        self.type = None
        self.history = None
        self.num_cs_type = None

    def initialize(self):
        user_type = self.generate_random_user_type()
        user_goal = self.generate_random_user_goal(user_type)

        self.goal = user_goal
        self.type = user_type
        self.agenda = self.generate_user_agenda()
        # 4 Types of users
        self.num_cs_type = [0, 0, 0, 0]

    @staticmethod
    def generate_random_user_goal(user_type):
        user_goal_slots = dialog_config.user_goal_slots
        user_goal = {}
        for slot in user_goal_slots:
            if len(user_goal_slots[slot]) == 2:
                user_goal[slot] = False
            if len(user_goal_slots[slot]) == 1:
                user_goal.update({slot: user_goal_slots[slot]})

        if user_type['num_person'] > 0:
            user_goal['goal_person'] = True

        if user_type['num_session'] > 0:
            user_goal['goal_session'] = True

        return user_goal

    @staticmethod
    def generate_random_user_type():
        num_reco_str = 'num_reco'
        primary_goal_str = 'primary_goal'
        time_str = 'time'
        rapport_care_str = 'rapport_care'
        cs_type_str = 'cs_type'

        user_type_slots = dialog_config.user_type_slots
        prob_user_type = dialog_config.prob_user_type

        user_type = {}
        binary_slots = []

        for slot in user_type_slots:
            if len(user_type_slots[slot]) == 2:
                binary_slots.append(slot)

        binary_slots.remove(rapport_care_str)

        # Pick a slot value according to the result of the biased coin toss
        for b_slot in binary_slots:
            r = random.random()
            if prob_user_type[b_slot] > r:
                user_type[b_slot] = True
            else:
                user_type[b_slot] = False

        # Select the total number of recos using a multinomial distribution
        x = np.random.multinomial(1, prob_user_type[num_reco_str])
        num_reco = int(np.where(x == 1)[0][0] + 1)
        user_type[num_reco_str] = num_reco
        # Select the number of person recos using a multinomial distribution
        x = np.random.multinomial(1, prob_user_type[num_reco_str + "_" + str(
            num_reco)])
        num_person = int(np.where(x == 1)[0][0])
        num_session = num_reco - num_person
        user_type['num_person'] = num_person
        user_type['num_session'] = num_session
        # Select the user CS type using a multinomial distribution
        x = np.random.multinomial(1, prob_user_type[cs_type_str])
        cs_type = int(np.where(x == 1)[0][0] + 1)
        user_type[cs_type_str] = cs_type

        # Primary Goal
        if num_person > 0:
            user_type[primary_goal_str] = 'goal_person'
        else:
            user_type[primary_goal_str] = 'goal_session'

        # Time at hand (Less / Enough / More)
        if num_reco <= 2:
            user_type[time_str] = user_type_slots[time_str][0]
        elif num_reco <= 4:
            user_type[time_str] = user_type_slots[time_str][1]
        else:
            user_type[time_str] = user_type_slots[time_str][2]

        # Does the user care about building rapport or not?
        r = random.random()
        if prob_user_type[rapport_care_str][num_reco-1] > r:
            user_type[rapport_care_str] = True
        else:
            user_type[rapport_care_str] = False

        small_penalty = dialog_config.small_penalty
        large_penalty = dialog_config.large_penalty

        # If the user cares about building rapport, it will apply a small per-
        # turn penalty only if it has less time. On the other hand, if the user
        # doesn't care about building rapport, it will apply a large per-turn
        # penalty if it has less time, small per-turn penalty if it has enough
        # time and no penalty if it has more time.

        if user_type[rapport_care_str]:
            if user_type[time_str] == 'LT':
                user_type['penalty'] = small_penalty
            else:
                user_type['penalty'] = 0
        else:
            if user_type[time_str] == 'LT':
                user_type['penalty'] = large_penalty
            elif user_type[time_str] == 'ET':
                user_type['penalty'] = small_penalty
            else:
                user_type['penalty'] = 0

        return user_type

    def generate_user_agenda(self):
        user_agenda = []
        user_action = {}
        pass
        user_action['act'] = 'bye'
        user_action['inform_slots'] = {}
        user_agenda.append(user_action)

        for slot in self.goal.keys():
            user_action = {}
            pass
            user_action['act'] = 'inform'
            user_action['inform_slots'] = {}
            user_action['inform_slots'][slot] = self.goal[slot]
            user_agenda.append(user_action)

        return user_agenda

    def next(self, agent_action, state):
        # update_goal(self, agent_action)
        self.update_agenda(agent_action, state)
        # print(self.agenda)
        return self.agenda.pop()

    def update_agenda(self, agent_action, state):
        user_action = {}
        pass
        user_action['act'] = 'null'
        user_action['inform_slots'] = {}
        user_action['request_slots'] = ''

        act = agent_action['act']
        CS = agent_action['CS']

        # Keep track of CS counts
        if CS == 'None':
            self.num_cs_type[0] += 1
        if CS == 'SD' or CS == 'QESD':
            self.num_cs_type[1] += 1
        if CS == 'SD' or CS == 'PR':
            self.num_cs_type[2] += 1
        if CS == 'PR' or CS == 'HE':
            self.num_cs_type[3] += 1

        if act == 'greeting':
            user_action['act'] = 'greeting'
            state['penalty'] = self.type['penalty']

        elif act == 'bye':
            user_action['act'] = 'bye'

        elif act == 'request':
            user_action['act'] = 'inform'
            user_action['inform_slots'] = self.process_request_act(
                agent_action, state)

        self.agenda.append(user_action)

    def process_request_act(self, agent_action, state):
        slot = agent_action['request_slots']
        decision_points = dialog_config.decision_points
        prob_funcs = dialog_config.prob_funcs

        if slot == 'selfie':
            return {slot: True}

        for t in self.type.keys():
            if slot == t:
                return {slot: self.type[t]}

        for g in self.goal.keys():
            if slot == g:
                return {slot: self.goal[g]}

        for dp in decision_points:
            if slot == 'feedback' and state['rapport_built'] is None:
                # Rapport built or not?
                if self.num_cs_type[self.type['cs_type']-1] >= \
                                dialog_config.threshold_cs[self.type[
                                    'cs_type']-1] * state['turn']:
                    state['rapport_built'] = True
                else:
                    state['rapport_built'] = False
            if slot == dp:
                val = prob_funcs[slot](self.type, state)
                return {slot: val}


if __name__ == "__main__":
    user_sim = RuleBasedUserSimulator()
    user_sim.initialize()

    # agent_action = {}
    # pass
    # agent_action['act'] = 'request'
    # agent_action['inform_slots'] = {}
    # agent_action['request_slots'] = 'feedback'
    # user_sim.update_agenda(agent_action)
