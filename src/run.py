from src.user_simulator.my_rule_based_user_sim import RuleBasedUserSimulator
from src.agent.rule_based_agent import RuleBasedAgent
from src.dialog_arbiter.arbiter import DialogArbiter


def main():
    num_dialogs = 1
    user = RuleBasedUserSimulator()
    agent = RuleBasedAgent()
    arbiter = DialogArbiter(user, agent)
    run_dialog(arbiter, num_dialogs)


def run_dialog(arbiter, num_dialogs):
    cumulative_reward = 0
    for dialog in range(num_dialogs):
        print("Dialog: " + str(dialog))
        arbiter.initialize_episode()
        episode_over = False
        while not episode_over:
            episode_over, reward = arbiter.next_step()
            if episode_over:
                cumulative_reward += reward


if __name__ == "__main__":
    main()
