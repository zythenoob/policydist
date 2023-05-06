import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from ablator import Results

from distributed import PDParallelConfig


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def get_display_name(name):
    if name == "pd":
        return "PD (FIFO)"
    elif name == "spd":
        return "SPD (FIFO)"
    elif name == "spdb":
        return "SPD (FIFO)"
    elif name == "onlinepd":
        return "Online PD (FIFO)"
    elif name == "reservoirpd":
        return "PD (Reservoir)"
    return ""


def plot_lines(path, ep, t_r, updates, rewards):

    fig = plt.figure(figsize=(7, 4))  # Create matplotlib figure
    plt.tight_layout()
    ax = fig.add_subplot(111)  # Create matplotlib axes
    # ax2 = ax.twinx()
    ax.axhline(y=t_r, color='r', linestyle='dotted', label="Teacher")
    # min_reward = 0

    for k in updates.keys():
        if len(updates[k][0]) == 0:
            continue
        name = k
        reward_mean = np.mean(rewards[name], axis=1)
        reward_std = np.std(rewards[name], axis=1)
        update_mean = np.mean(updates[name], axis=1)
        # min_reward = min(min_reward, min(update_mean))
        # rewards
        ax.plot(ep, reward_mean, label=get_display_name(name))
        ax.fill_between(ep, reward_mean-reward_std, reward_mean+reward_std, alpha=.1)
        # updates
        # ax2.plot(ep, update_mean, linestyle='dashed', label=f'{name} updates')

    ax.set_ylabel('Average Reward', fontsize=12)
    # ax2.set_ylabel('Num updates', fontsize=12)
    ax.set_xlabel('Teacher episodes', fontsize=12)

    lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines, labels, loc='best')
    # ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # ax.set_xticks(np.arange(8), labels, rotation=0)
    ax.set_xlim((0, max_ep))
    # ax.set_xlim((0, 300))
    ax.set_ylim((0, t_r + 100))
    fig.tight_layout(pad=0.4)
    img = fig2img(fig)
    plt.close()
    path.mkdir(exist_ok=True, parents=True)
    img_path = path.joinpath(f"result_{env_name}.png")
    img.save(img_path)


if __name__ == "__main__":
    # env_name = "walker"
    # env_name = "hopper"
    env_name = "halfcheetah"
    experiment_dir = Path(f"/home/ji/experiments/{env_name}")
    experiment_paths = list(experiment_dir.rglob("default_config.yaml"))
    results_dir = Path("/home/ji/experiments/").joinpath("analysis")

    max_ep = 200

    # fig, ax = plt.subplots()
    episodes = np.arange(max_ep) + 1
    updates = {}
    rewards = {}
    teacher_reward = []

    for path in experiment_paths:
        if 'experiment_fa40_04af' in path.as_posix():
            continue
        path = path.parent
        try:

            results = Results(
                config=PDParallelConfig,
                experiment_dir=path,
            )
            method_name = results.config.model_config.name
            print(method_name)
            data = results.data

            if method_name not in updates:
                updates[method_name] = [[] for _ in range(max_ep)]
                rewards[method_name] = [[] for _ in range(max_ep)]

            for i, row in data.iterrows():
                if i >= max_ep:
                    continue
                train_ep = int(row['current_epoch'])
                updates[method_name][train_ep].append(float(row['current_iteration']))
                rewards[method_name][train_ep].append(float(row['val_reward']))
                teacher_reward.append(float(row['train_reward']))

        except Exception as e:
            traceback.print_exc()
            raise e
        
    teacher_reward = np.mean(teacher_reward)

    plot_lines(results_dir, episodes, teacher_reward, updates, rewards)