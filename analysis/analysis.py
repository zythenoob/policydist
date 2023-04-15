import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trainer.analysis.plot import CustomReport
from trainer.analysis.utils import fig2img

from distributed import ParallelConfig


def plot_lines(path, ep, t_r, updates, rewards):

    fig = plt.figure(figsize=(7, 4))  # Create matplotlib figure
    plt.tight_layout()
    ax = fig.add_subplot(111)  # Create matplotlib axes
    ax2 = ax.twinx()
    ax.axhline(y=t_r, color='r', linestyle='dotted', label="Teacher reward")

    for k in updates.keys():
        if len(updates[k][0]) == 0:
            continue
        name = k
        reward_mean = np.mean(rewards[name], axis=1)
        reward_std = np.std(rewards[name], axis=1)
        update_mean = np.mean(updates[name], axis=1)
        # rewards
        ax.plot(ep, reward_mean, label=f'{name} rewards')
        ax.fill_between(ep, reward_mean-reward_std, reward_mean+reward_std, alpha=.1)
        # updates
        ax2.plot(ep, update_mean, linestyle='dashed', label=f'{name} updates')

    ax.set_ylabel('Average Reward', fontsize=12)
    ax2.set_ylabel('Num updates', fontsize=12)
    ax.set_xlabel('Teacher episodes')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower right')

    # ax.set_xticks(np.arange(8), labels, rotation=0)
    ax.set_xlim((0, 100))
    fig.tight_layout(pad=0.4)
    img = fig2img(fig)
    plt.close()
    path.mkdir(exist_ok=True, parents=True)
    img_path = path.joinpath(f"result_walker.png")
    img.save(img_path)


if __name__ == "__main__":
    # spd_experiment_dir = Path("/home/ji/experiments/hopper_spd/mp_run_3456_8ff5")
    # pd_experiment_dir = Path("/home/ji/experiments/hopper_pd/mp_run_dcb9_a891")
    experiment_dir = Path("/home/ji/experiments/walker")
    # experiment_dir = Path("/home/ji/experiments/mp_run_d9a2_4a5b")
    run_configs = list(experiment_dir.rglob("run_config.yaml"))
    results_dir = Path("/home/ji/experiments/").joinpath("analysis")

    # fig, ax = plt.subplots()
    episodes = np.arange(100) + 1
    updates = {}
    rewards = {}
    teacher_reward = []

    for run_config in run_configs:
        if 'mp_run_6d4f_8a13' in run_config.as_posix():
            continue
        config = ParallelConfig.load(run_config)
        method_name = config.model_config.name
        try:
            results = CustomReport(
                config=config,
                parse_informative=False,
            )

            results = CustomReport(
                config=config,
                parse_informative=False,
            ).results

            updates[method_name] = [[] for _ in range(100)]
            rewards[method_name] = [[] for _ in range(100)]

            for i, row in results.iterrows():
                train_ep = int(row['train_aux_traj_names'])
                updates[method_name][train_ep].append(float(row['num_updates']))
                rewards[method_name][train_ep].append(float(row['val_reward']))
                teacher_reward.append(float(row['train_reward']))

        except Exception as e:
            traceback.print_exc()
            raise e
        
    teacher_reward = np.mean(teacher_reward)

    plot_lines(results_dir, episodes, teacher_reward, updates, rewards)