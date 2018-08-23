import dialog_config
import numpy as np


def clip(rapp):
    if rapp < dialog_config.rapp_min:
        return dialog_config.rapp_min
    elif rapp > dialog_config.rapp_max:
        return dialog_config.rapp_max
    else:
        return rapp.data.cpu().numpy()


def get_cs(enc, thresh, prob_pred):
    # y_pred = prob_pred[0].copy()
    # y_pred[y_pred >= thresh] = 1
    # y_pred[y_pred < thresh] = 0
    # if np.sum(y_pred) == 0:
    #     y_pred[np.argmax(prob_pred[0])] = 1

    y_pred = np.random.binomial(1, prob_pred[0])
    # print("Probability vector: ", prob_pred[0])
    # print("Prediction: ", y_pred)
    if np.sum(y_pred) == 0:
        y_pred = prob_pred[0].copy()
        y_pred[y_pred >= thresh] = 1
        y_pred[y_pred < thresh] = 0

        if np.sum(y_pred) == 0:
            y_pred[np.argmax(prob_pred[0])] = 1

    return y_pred, enc.unfit(y_pred)


def get_agent_intention(agent_action):
    act = agent_action['act']
    req_slots = agent_action['request_slots']
    inf_slots = agent_action['inform_slots']

    if req_slots == '' and inf_slots == {}:
        return act + '()'
    elif req_slots != '' and inf_slots == {}:
        return act + '(' + modify_req_slot(req_slots) + ')'
    else:
        return act + '(' + modify_inf_slot(inf_slots) + ')'


def modify_req_slot(req_slots):
    if req_slots in ['goal_person', 'goal_session', 'goal_food']:
        return 'goal'
    else:
        return req_slots


def modify_inf_slot(inf_slots):
    islot = list(inf_slots.values())[0]
    if islot in ['info_person', 'info_session']:
        return 'info'
    else:
        return islot
