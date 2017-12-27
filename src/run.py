from src.user_simulator.my_rule_based_user_sim import RuleBasedUserSimulator
from src.agent.rule_based_agent import RuleBasedAgent
from src.dialog_arbiter.arbiter import DialogArbiter
from src import dialog_config


def main():
    num_dialogs = 10

    goal_type = "random"
    user = RuleBasedUserSimulator(goal_type)
    agent = RuleBasedAgent()
    slot_set = dialog_config.slot_set
    arbiter = DialogArbiter(user, agent)
    run_dialog(arbiter, num_dialogs, slot_set)


def run_dialog(arbiter, num_dialogs, slot_set):
    cumulative_reward = 0
    count_slots = dialog_config.count_slots
    reward_slots = dialog_config.reward.keys()

    for dialog in range(num_dialogs):
        print("Dialog: " + str(dialog))
        arbiter.initialize(slot_set)
        dialog_over = False
        while not dialog_over:
            reward, dialog_over, state = arbiter.next()

        print()
        print("STATS:")
        print("Reward: " + str(reward))
        print()
        print("Turns: " + str(state['turn'] / 2 - 1))
        print()

        for c_slot in count_slots:
            print("Total " + c_slot + " recommendations: ", end="")
            print(state[c_slot])
            for r_slot in reward_slots:
                if r_slot == 'feedback':
                    print("Feedback good count: " + str(state['num_accepted'][
                                                            c_slot][r_slot]))
                else:
                    print("Top-link message count: " + str(
                        state['num_accepted'][c_slot][r_slot]))
            print()


if __name__ == "__main__":
    main()
