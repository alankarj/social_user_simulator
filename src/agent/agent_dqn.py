import numpy as np
import random
from src.agent.rule_based_agent import RuleBasedAgent
from src.dqn.simple_dqn import SimpleDQN
import math
import json
import tensorflow as tf

console_width = 137  # Used for printing phase string


class AgentDQN:
    def __init__(self, params=None):
        self.slot_set = params['slot_set']
        self.user_act_set = params['user_act_set']
        self.agent_act_set = params['agent_act_set']
        self.user_cs_set = params['user_cs_set']
        self.agent_cs_set = params['agent_cs_set']
        self.slot_set = params['slot_set']
        self.phase_set = params['phase_set']
        self.feasible_actions = params['feasible_actions']
        self.bool_slots = params['bool_slots']
        self.max_turn = params['max_turns']
        self.reward_slots = params['reward_slots']
        self.max_recos = params['max_recos']

        self.num_actions = len(self.feasible_actions)

        self.user_act_cardinality = len(self.user_act_set.keys())
        self.agent_act_cardinality = len(self.agent_act_set.keys())

        self.user_cs_cardinality = len(self.user_cs_set.keys())
        self.agent_cs_cardinality = len(self.agent_cs_set.keys())

        self.slot_set_cardinality = len(self.slot_set.keys())
        self.phase_set_cardinality = len(self.phase_set.keys())

        self.bool_slot_cardinality = len(self.bool_slots.keys())

        self.fillable_slots = [k for (k, v) in self.slot_set.items() if v <=
                               10]
        self.num_fillable_slots = len(self.fillable_slots)

        self.count_slots = [k for (k, v) in self.slot_set.items() if 10 < v <
                            13]
        self.num_count_slots = len(self.count_slots)
        self.num_rewards_slots = len(self.reward_slots)

        self.count_slots_dim = (self.max_recos + 2) * self.num_count_slots
        self.num_accepted_dim = self.num_count_slots * self.num_rewards_slots \
                                * self.max_recos

        self.state_dim = self.user_act_cardinality + 1 + \
                         self.slot_set_cardinality + 1 + \
                         self.num_fillable_slots + self.count_slots_dim + \
                         self.max_turn + self.num_accepted_dim + \
                         self.agent_act_cardinality + 1 +  2 * (
            self.slot_set_cardinality + 1) + self.phase_set_cardinality + 1 +\
                         self.user_cs_cardinality + 1 + \
                         self.agent_cs_cardinality + 1 + self.bool_slot_cardinality

        # print("State Dimension: ", self.state_dim)
        # print("Action Size: ", self.num_actions)

        self.experience_replay_pool_size_SL = params['ERP_size_SL']
        self.experience_replay_pool_size_RL = params['ERP_size_RL']

        self.hidden_dim = params['hidden_dim']
        self.gamma = params['gamma']
        self.predict_mode = params['predict_mode']
        self.warm_start = params['warm_start']

        # Keep bootstrapping data in the following pool
        self.erp_sl = {'data': [], 'priority': []}
        # Keep agent's own data in the following pool
        self.erp_rl = {'data': [], 'priority': []}


        self.representation = None
        self.action = None
        self.rule_based_agent = RuleBasedAgent()
        self.gamma = params['gamma']

        self.dqn = SimpleDQN(self.state_dim, self.num_actions,
                             self.hidden_dim, self.gamma)

        self.epsilon = params['epsilon']
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95

        self.curr_phase = None
        self.prev_phase = None

        # Parameters for DQN training
        self.alpha = 0.4
        self.epsilon_sl = 1
        self.epsilon_rl = 0.001

    def initialize(self, state):
        if self.warm_start == 1:
            return self.rule_based_agent.initialize()
        else:
            return self.next(state)

    def next(self, state):
        """ DQN: Input state, output action """
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(state, self.representation)
        final_action = self.feasible_actions[self.action]

        # if self.warm_start == 2:
        #     self.curr_phase = final_action['phase']
        #     if self.curr_phase != self.prev_phase:
        #         self.print_phase()
        #     self.prev_phase = self.curr_phase

        return final_action

    def run_policy(self, state, representation):
        """ epsilon-greedy policy """
        if self.warm_start == 1:
            if len(self.experience_replay_pool) >= \
                    self.experience_replay_pool_size:
                self.warm_start = 2
            return self.rule_policy(state)

        else:
            if random.random() < self.epsilon:
                return random.randint(0, self.num_actions - 1)
            else:
                return self.dqn.get_best_action(representation)

    def rule_policy(self, state):
        act_slot_response = self.rule_based_agent.next(state)
        return self.action_index(act_slot_response)

    def action_index(self, act_slot_response):
        """ Return the index of action """
        for i, action in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i

    def prepare_state_representation(self, state):
        # print("State: ", json.dumps(state, indent=2))
        """ Create the representation for each state """
        if 'user_action' in state.keys():
            bool_user = 1
            user_action = state['user_action']
        else:
            bool_user = 0

        if 'agent_action' in state.keys():
            bool_agent = 1
            agent_action = state['agent_action']
        else:
            bool_agent = 0

        # User action representation
        user_act_rep = np.zeros((1, self.user_act_cardinality + 1))
        if bool_user == 0:
            user_act_rep[0, self.user_act_cardinality] = 1.0
        elif bool_user == 1:
            user_act_rep[0, self.user_act_set[user_action['act']]] = 1.0

        # print("User act rep: ", user_act_rep)

        # User CS representation
        user_cs_rep = np.zeros((1, self.user_cs_cardinality + 1))
        if bool_user == 0:
            user_cs_rep[0, self.user_cs_cardinality] = 1.0
        elif bool_user == 1:
            user_cs_rep[0, self.user_cs_set[user_action['CS']]] = 1.0

        # print("User CS rep: ", user_cs_rep)

        # Inform slot representation
        user_inform_slots_rep = np.zeros((1, self.slot_set_cardinality + 1))
        if bool_user == 0:
            user_inform_slots_rep[0, self.slot_set_cardinality] = 1.0
        elif bool_user == 1:
            for slot in user_action['inform_slots'].keys():
                user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        # print("User Inform Slots rep: ", user_inform_slots_rep)

        # Bag of bool slots representation
        bool_slots_rep = np.zeros((1, self.bool_slot_cardinality))
        for slot in self.bool_slots:
            if slot == 'primary_goal':
                if state[slot] == 'goal_person':
                    bool_slots_rep[0, self.bool_slots[slot]] = 1.0
            else:
                if state[slot]:
                    bool_slots_rep[0, self.bool_slots[slot]] = 1.0

        # print("Bool Slots rep: ", bool_slots_rep)

        # Bag of fillable slots representation
        fillable_slots_rep = np.zeros((1, self.num_fillable_slots))
        for slot in self.fillable_slots:
            if state[slot] != '':
                fillable_slots_rep[0, self.slot_set[slot]] = 1.0

        # print("Fillable Slots rep: ", fillable_slots_rep)

        # Bag of count slots representation
        count_slots_rep = np.zeros((1, self.count_slots_dim))
        index = 0
        for slot in self.count_slots:
            count_slots_rep[0, index + state[slot]] = 1.0
            index += self.max_recos

        # print("Count Slots rep: ", count_slots_rep)

        # Turn count representation
        turn_rep = np.zeros((1, self.max_turn))
        turn_rep[0, int(state['turn']*0.5)] = 1.0

        # print("Turn rep: ", turn_rep)

        # num_accepted representation
        num_accepted_rep = np.zeros((1, self.num_accepted_dim))
        index = 0
        for c_slot in self.count_slots:
            for r_slot in self.reward_slots:
                num_accepted_rep[0, index + state['num_accepted'][c_slot][
                    r_slot]] = 1.0
                index += self.max_recos

        # print("Num accepted rep: ", num_accepted_rep)

        # Agent action representation
        agent_act_rep = np.zeros((1, self.agent_act_cardinality + 1))
        if bool_agent == 0:
            agent_act_rep[0, self.agent_act_cardinality] = 1.0
        elif bool_agent == 1:
            agent_act_rep[0, self.agent_act_set[agent_action['act']]] = 1.0

        # print("Agent act rep: ", agent_act_rep)

        # Agent Inform slot representation
        agent_inform_slots_rep = np.zeros((1, self.slot_set_cardinality + 1))
        if bool_agent == 0:
            agent_inform_slots_rep[0, self.slot_set_cardinality] = 1.0
        elif bool_agent == 1:
            for slot in agent_action['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        # print("Agent inform slots rep: ", agent_inform_slots_rep)

        # Agent Request slot representation
        agent_request_slots_rep = np.zeros((1, self.slot_set_cardinality + 1))
        if bool_agent == 0:
            agent_request_slots_rep[0, self.slot_set_cardinality] = 1.0
        elif bool_agent == 1:
            request_slot = agent_action['request_slots']
            if request_slot is not '':
                agent_request_slots_rep[0, self.slot_set[request_slot]] = 1.0

        # print("Agent request slots rep: ", agent_request_slots_rep)

        # Phase representation
        phase_rep = np.zeros((1, self.phase_set_cardinality + 1))
        if bool_agent == 0:
            phase_rep[0, self.phase_set_cardinality] = 1.0
        elif bool_agent == 1:
            phase_rep[0, self.phase_set[state['phase']]] = 1.0

        # print("Agent phase rep: ", phase_rep)

        # Agent CS representation
        agent_cs_rep = np.zeros((1, self.agent_cs_cardinality + 1))
        if bool_agent == 0:
            agent_cs_rep[0, self.agent_cs_cardinality] = 1.0
        elif bool_agent == 1:
            agent_cs_rep[0, self.agent_cs_set[agent_action['CS']]] = 1.0

        # print("Agent CS rep: ", agent_cs_rep)

        full_state_rep = np.hstack([user_act_rep, user_inform_slots_rep,
                                    fillable_slots_rep, count_slots_rep,
                                    turn_rep, num_accepted_rep,
                                    agent_act_rep, agent_inform_slots_rep,
                                    agent_request_slots_rep, phase_rep,
                                    user_cs_rep, agent_cs_rep, bool_slots_rep])

        # print("State representation: ", full_state_rep)
        return full_state_rep

    # Sample transitions from the experience replay buffer based on priority
    def get_training_data(self, batch_size):
        priority_list = self.erp_sl['priority'] + self.erp_rl['priority']
        assert priority_list != []
        num = np.array(priority_list)**self.alpha
        den = np.sum(num)
        p = num/den
        _, indices = np.where(np.random.multinomial(1, p, batch_size) == 1)
        ind_sl = indices[indices <= len(self.erp_sl)]
        ind_rl = indices[indices > len(self.erp_sl)] - len(self.erp_sl)
        if ind_sl.shape[0] != 0:
            samples_sl = list(np.take(self.erp_sl['data'], ind_sl))
        if ind_rl.shape[0] != 0:
            samples_rl = list(np.take(self.erp_rl['data'], ind_rl))
        return samples_sl + samples_rl, ind_sl, ind_rl

    def register_experience_replay_tuple(self, s_t, a_t, r_t, s_tplus1,
                                             dialog_over):
        """ Register feedback from the environment, to be stored as
        future training data """

        s_t_rep = self.prepare_state_representation(s_t)
        s_tplus1_rep = self.prepare_state_representation(s_tplus1)

        training_example = (s_t_rep, self.action_index(a_t), r_t, s_tplus1_rep,
                            dialog_over)

        if self.warm_start == 1:
            self.erp_sl['data'].append(training_example)
        else:
            self.erp_rl['data'].append(training_example)

    def train(self, batch_size):
        sample, ind_sl, ind_rl = self.get_training_data(batch_size)
        loss, td_error = self.dqn.train(sample)
        print("Mean squared loss: ", loss)
        return loss

    def update_priority(self, ind_sl, ind_rl, td_error):
        assert len(td_error) == len(ind_sl) + len(ind_rl)
        td_error_sl = td_error[:len(ind_sl)]
        td_error_rl = td_error[len(ind_sl):]
        priority_sl = [delta + self.epsilon_sl for delta in td_error_sl]
        priority_rl = [delta + self.epsilon_rl for delta in td_error_rl]
        for i, ind in enumerate(ind_sl):
            self.erp_sl['priority'][ind] = priority_sl[i]
        for i, ind in enumerate(ind_rl):
            self.erp_rl['priority'][ind] = priority_rl[i]

    def update_target_network(self):
        self.dqn.update_target_q_network()

    def print_phase(self, phase):
        print(phase.center(console_width, '-'))