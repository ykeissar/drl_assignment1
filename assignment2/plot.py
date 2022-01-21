import matplotlib.pyplot as plt
import numpy as np
import json


def plot_losses():
    with open('losses/pg.json') as f:
        org = json.load(f)

    with open('losses/rein_p.json') as f:
        rein = json.load(f)

    with open('losses/ac_p.json') as f:
        ac = json.load(f)

    org = [float(p) for p in org]
    rein = [float(p) for p in rein]
    ac = [float(p) for p in ac]
    rate = 1000
    org, rein, ac = smooth(org, rate), smooth(rein, rate), smooth(ac, rate)

    plt.plot(np.arange(len(org))*rate, org, label='Policy Gradient')
    plt.plot(np.arange(len(rein))*rate, rein, label='REINFORCE')
    plt.plot(np.arange(len(ac))*rate, ac, label='Actor Critic')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Losses over iterations')
    plt.legend()
    plt.show()


def smooth(a, rate):
    diff = rate - (len(a) % rate)
    a += (list(np.ones((diff)) * a[-1]))
    b = np.array(a)

    b = b.reshape(-1, rate)
    b = np.mean(b, axis=1)
    return b

if __name__ == '__main__':
    plot_losses()
    # smooth([],50)