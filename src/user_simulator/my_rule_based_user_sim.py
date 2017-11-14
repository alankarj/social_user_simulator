from src import dialog_config
import random


class RuleBasedUserSimulator:
    def __init__(self, goal_type):
        # goal is a dict containing inform slots and request slots
        self.goal = None
        # agenda is a stack representing pending user actions
        self.agenda = None
        self.type = None
        self.history = None
        self.goal_type = goal_type

    def initialize(self):
        if self.goal_type == 'fixed':
            user_goal = self.generate_fixed_user_goal()
            user_type = self.generate_fixed_user_type(user_goal)
        elif self.goal_type == 'random':
            user_goal = self.generate_random_user_goal()
            user_type = self.generate_random_user_type(user_goal)

        self.goal = user_goal
        self.type = user_type
        self.agenda = self.generate_user_agenda()
        self.history = []

        #print("user goal: ")
        #print(self.goal)
        #print("user type: ")
        #print(self.type)
        #print("user agenda: ")
        #print(self.agenda)

    @staticmethod
    def generate_fixed_user_goal():
        user_goal = dialog_config.fixed_user_goal
        return user_goal

    @staticmethod
    def generate_fixed_user_type():
        user_type = dialog_config.fixed_user_type
        return user_type

    @staticmethod
    def generate_random_user_goal():
        user_goal_slots = dialog_config.user_goal_slots
        user_goal = {}
        binary_goal_slots = []
        for slot in user_goal_slots:
            if len(user_goal_slots[slot]) == 2:
                binary_goal_slots.append(slot)
            elif len(user_goal_slots[slot]) == 1:
                user_goal.update({slot: user_goal_slots[slot]})

        prob_user_goal = dialog_config.prob_user_goal

        # Pick a slot value according to the result of the biased coin toss
        for slot in binary_goal_slots:
            r = random.random()
            if prob_user_goal[slot] > r:
                user_goal[slot] = True
            else:
                user_goal[slot] = False

        for slot in user_goal_slots:
            if len(user_goal_slots[slot]) == 1:
                user_goal.update({slot: user_goal_slots[slot]})

        return user_goal

    @staticmethod
    def generate_random_user_type(user_goal):
        user_type_slots = dialog_config.user_type_slots
        user_goal_slots = dialog_config.user_goal_slots
        prob_user_type = dialog_config.prob_user_type

        user_type = {}
        binary_slots = []
        binary_goal_slots = []
        slots_true = []
        # Assumption: There exists only a single non_binary_slot which
        # corresponds to a selection from all the binary goal slots which
        # are True
        non_binary_slot = []

        for slot in user_type_slots:
            if len(user_type_slots[slot]) == 2:
                binary_slots.append(slot)
            elif len(user_type_slots[slot]) > 2:
                non_binary_slot.append(slot)

        # Pick a slot value according to the result of the biased coin toss
        for b_slot in binary_slots:
            r = random.random()
            if prob_user_type[b_slot] > r:
                user_type[b_slot] = True
            else:
                user_type[b_slot] = False

        # Determine which slots have True/False values in user_goal_slots
        for slot in user_goal_slots:
            if len(user_goal_slots[slot]) == 2:
                binary_goal_slots.append(slot)

        # Determine which slots are True in binary_goal_slots
        for slot in binary_goal_slots:
            if user_goal[slot]:
                slots_true.append(slot)

        # Pick one of the slots with True value
        for nb_slot in non_binary_slot:
            i = random.randint(0, len(slots_true) - 1)
            user_type[nb_slot] = slots_true[i]

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

    def sample_user_action(self):
        user_action = self.agenda.pop()
        self.history.add(user_action)
        return self.agenda.pop()

    def next(self, agent_action):
        # update_goal(self, agent_action)
        self.update_agenda(agent_action)
        # print(self.agenda)
        return self.agenda.pop()

    def update_agenda(self, agent_action):
        user_action = {}
        pass
        user_action['act'] = 'null'
        user_action['inform_slots'] = {}
        user_action['request_slots'] = ''

        act = agent_action['act']

        if act == 'greeting':
            user_action['act'] = 'greeting'

        elif act == 'bye':
            user_action['act'] = 'bye'

        elif act == 'request':
            user_action['act'] = 'inform'
            user_action['inform_slots'] = self.process_request_act(agent_action)

        #else:
        #    user_action['act'] = ''

        self.agenda.append(user_action)
        #print(user_action)
        #dialog_config.print_info(user_action)

    def process_request_act(self, agent_action):
        slot = agent_action['request_slots']
        threshold = dialog_config.threshold
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
            if slot == dp:
                prob = prob_funcs[slot](self.history)
                if prob > threshold[slot]:
                    return {slot: True}
                else:
                    return {slot: False}

if __name__ == "__main__":
    goal_type = "random"
    user_sim = RuleBasedUserSimulator(goal_type)

    user_sim.initialize()

    agent_action = {}
    pass
    agent_action['act'] = 'request'
    agent_action['inform_slots'] = {}
    agent_action['request_slots'] = 'feedback'
    user_sim.update_agenda(agent_action)
