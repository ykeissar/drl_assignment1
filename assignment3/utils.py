import ujson as json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

ENVS = {'cp': 'CartPole-v1', 'acro': 'Acrobot-v1', 'mcc': 'MountainCarContinuous-v0'}


def save_losses(curr_ep, v_loss, p_loss, env_name, label, time):
    d = {'episode_solved': curr_ep, 'v_loss': [str(l) for l in v_loss], 'p_loss': [str(l) for l in p_loss],
         'label': label, 'time':str(time)}
    with open(f'losses/{env_name}_{datetime.now().strftime("%d-%m-%Y_%H:%M")}.json', 'w+') as f:
        json.dump(d, f)


def plot_losses(jsons, rate=1000):
    ob = {'labels': [], 'episodes': [], 'iterations': [], 'times':[]}
    for file in jsons:
        with open(file) as f:
            d = json.load(f)

        ob['labels'].append(d['label'])
        ob['episodes'].append(d['episode_solved'])
        ob['iterations'].append(len(d['p_loss']))
        ob['times'].append(d['time'])

        p_loss = [float(p) for p in d['p_loss']]
        p_loss = smooth(p_loss, rate)
        plt.plot(np.arange(len(p_loss)) * rate, p_loss, label=d['label'])

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Losses over iterations')
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


if __name__ == '__main__':

    plot_losses(['losses/CartPole-v1_target_31-01-2022_17:14.json'], rate=100)
