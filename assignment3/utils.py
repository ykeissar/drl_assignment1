import ujson as json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

ENVS = {'cp': 'CartPole-v1', 'acro': 'Acrobot-v1', 'mcc': 'MountainCarContinuous-v0'}


def save_losses(curr_ep, v_loss, p_loss, env_name, label, time, ep_rew):
    d = {'episode_solved': curr_ep, 'v_loss': [str(l) for l in v_loss], 'p_loss': [str(l) for l in p_loss],
         'label': label, 'time':str(time), 'ep_rew': [str(r) for r in ep_rew]}
    with open(f'losses/{env_name}_{datetime.now().strftime("%d-%m-%Y_%H:%M")}.json', 'w+') as f:
        json.dump(d, f)


def plot_losses(jsons, title, rate=1000):
    ob = {'labels': [], 'episodes': [], 'iterations': [], 'times':[]}
    jsons = [json.load(open(file)) for file in jsons]

    for d in jsons:
        ob['labels'].append(d['label'])
        ob['episodes'].append(d['episode_solved'])
        ob['iterations'].append(len(d['p_loss']))
        ob['times'].append(d['time'])
        p_loss = [float(p) for p in d['p_loss']]
        p_loss = smooth(p_loss, rate)
        plt.plot(np.arange(len(p_loss)) * rate, p_loss, label=d['label'])

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'{title} - Losses')
    plt.legend()
    plt.show()

    for d in jsons:
        ep_rew = [float(r) for r in d['ep_rew']]
        ep_rew = ep_rew[:d['episode_solved']+1]

        plt.plot(np.arange(len(ep_rew)), ep_rew, label=d['label'])

    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'{title} - Rewards')
    plt.legend()
    plt.show()

    for i in range(len(jsons)):
        print(f'{ob["labels"][i]}: {ob["episodes"][i]} episodes, {ob["iterations"][i]} iterations, {ob["times"][i]} time')


def smooth(a, rate):
    diff = rate - (len(a) % rate)
    a += (list(np.ones((diff)) * a[-1]))
    b = np.array(a)

    b = b.reshape(-1, rate)
    b = np.mean(b, axis=1)
    return b


def get_args(env):
    args = {}
    if env == ENVS['mcc']:
        args['learning_rate'] = 0.00001
        args['v_learning_rate'] = 0.0005
        args['discount_factor'] = 0.999
        args['epsilon'] = 0.5

    return args

if __name__ == '__main__':
    a = ['losses/MountainCarContinuous-v0_04-02-2022_17:08.json', 'losses/MountainCarContinuous-v0_target_04-02-2022_17:47.json', 'losses/MountainCarContinuous-v0_target_pnn_05-02-2022_19:33.json']
    # a = ['losses/CartPole-v1_01-02-2022_21:18.json', 'losses/CartPole-v1_target_02-02-2022_11:28.json', 'losses/CartPole-v1_target_pnn_04-02-2022_20:14.json']
    plot_losses(a, title='Transfer (from Cartpole) vs AC vs PNN', rate=100)
