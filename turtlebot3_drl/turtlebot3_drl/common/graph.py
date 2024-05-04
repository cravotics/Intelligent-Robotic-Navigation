import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from turtlebot3_drl.drl_environment.reward import SUCCESS
from .settings import GRAPH_DRAW_INTERVAL, GRAPH_AVERAGE_REWARD
from matplotlib.ticker import MaxNLocator

matplotlib.use('TkAgg')

class Graph():
    def __init__(self):
        plt.show()

        self.session_dir = ""
        self.legend_labels = ['Unknown', 'Success', 'Wall Collision', 'Dynamic Collision', 'Timeout', 'Tumble']
        self.legend_colors = ['b', 'g', 'r', 'c', 'm', 'y']

        self.outcome_histories = []

        self.global_steps = 0
        self.data_outcome_history = []
        self.data_rewards = []
        self.data_loss_critic = []
        self.data_loss_actor = []
        self.graphdata = [self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor]

        # Create a figure with subplots arranged differently
        self.fig = plt.figure(figsize=(20, 15))
        self.ax = [
            plt.subplot2grid((3, 2), (0, 0), colspan=2),
            plt.subplot2grid((3, 2), (1, 0)),
            plt.subplot2grid((3, 2), (1, 1)),
            plt.subplot2grid((3, 2), (2, 0)),
            plt.subplot2grid((3, 2), (2, 1))
        ]

        titles = ['Performance', 'Avg Actor Loss over Episode', 'Avg Reward over 10 Episodes']
        labels = [
            ('Episode', 'Outcomes Count'),
            ('Episode', 'Actor Loss'),
            ('Episode', 'Average Reward')
        ]

        for i, ax in enumerate(self.ax[:3]):
            ax.set_title(titles[i])
            ax.set_xlabel(labels[i][0])
            ax.set_ylabel(labels[i][1])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Remove empty subplots
        self.fig.delaxes(self.ax[3])
        self.fig.delaxes(self.ax[4])
        self.legend_set = False

    def set_graphdata(self, graphdata, episode):
        self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor = [graphdata[i] for i in range(len(self.graphdata))]
        self.graphdata = [self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor]
        self.draw_plots(episode)
        return self.global_steps

    def update_data(self, step, global_steps, outcome, reward_sum, loss_critic_sum, loss_actor_sum):
        self.global_steps = global_steps
        self.data_outcome_history.append(outcome)
        self.data_rewards.append(reward_sum)
        self.data_loss_critic.append(loss_critic_sum / step)
        self.data_loss_actor.append(loss_actor_sum / step)
        self.graphdata = [self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor]

    def draw_plots(self, episode):
        xaxis = np.array(range(1, episode + 1))

        # Plot outcome history
        for idx in range(len(self.data_outcome_history)):
            if idx == 0:
                self.outcome_histories = [[0], [0], [0], [0], [0], [0]]
                self.outcome_histories[self.data_outcome_history[0]][0] += 1
            else:
                for outcome_history in self.outcome_histories:
                    outcome_history.append(outcome_history[-1])
                self.outcome_histories[self.data_outcome_history[idx]][-1] += 1

        if len(self.data_outcome_history) > 0:
            i = 0
            for outcome_history in self.outcome_histories:
                self.ax[0].plot(xaxis, outcome_history, color=self.legend_colors[i], label=self.legend_labels[i])
                i += 1
            if not self.legend_set:
                self.ax[0].legend()
                self.legend_set = True

        # Plot actor loss
        y = np.array(self.data_loss_actor)
        self.ax[1].plot(xaxis, y)

        # Plot average reward
        count = int(episode / GRAPH_AVERAGE_REWARD)
        if count > 0:
            xaxis_new = np.array(range(GRAPH_AVERAGE_REWARD, episode + 1, GRAPH_AVERAGE_REWARD))
            averages = []
            for i in range(count):
                avg_sum = 0
                for j in range(GRAPH_AVERAGE_REWARD):
                    avg_sum += self.data_rewards[i * GRAPH_AVERAGE_REWARD + j]
                averages.append(avg_sum / GRAPH_AVERAGE_REWARD)
            y = np.array(averages)
            self.ax[2].plot(xaxis_new, y)

        plt.draw()
        plt.pause(0.2)
        plt.savefig(os.path.join(self.session_dir, "_figure.png"))

    def get_success_count(self):
        suc = self.data_outcome_history[-GRAPH_DRAW_INTERVAL:]
        return suc.count(SUCCESS)

    def get_reward_average(self):
        rew = self.data_rewards[-GRAPH_DRAW_INTERVAL:]
        return sum(rew) / len(rew)
