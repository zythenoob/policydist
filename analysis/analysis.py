import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trainer.analysis.plot import CustomReport
from trainer.analysis.utils import fig2img

from distributed import ParallelConfig


def plot_lines(path, ep, t_r, spd_r, spd_u, pd_r, pd_u):
    spd_r_mean = np.mean(spd_r, axis=1)
    spd_r_std = np.std(spd_r, axis=1)
    pd_r_mean = np.mean(pd_r, axis=1)
    pd_r_std = np.std(pd_r, axis=1)
    df = pd.DataFrame({
        'spd_r_mean': spd_r_mean, 'spd_r_std': spd_r_std, 'spd_u': spd_u, 
        'pd_r_mean': pd_r_mean, 'pd_r_std': pd_r_std, 'pd_u': pd_u, 
    }, index=ep)
    print(df)
    fig = plt.figure(figsize=(7, 4))  # Create matplotlib figure
    plt.tight_layout()
    ax = fig.add_subplot(111)  # Create matplotlib axes
    ax2 = ax.twinx()
    ax.axhline(y=t_r, color='r', linestyle='dotted', label="Teacher reward")

    df.spd_u.plot(kind='line', color='coral', ax=ax2, label="SPD updates")
    df.pd_u.plot(kind='line', color='red', ax=ax2, label="PD updates")
    df.spd_r_mean.plot(kind='line', ax=ax, label="SPD reward")
    df.pd_r_mean.plot(kind='line', color='green', ax=ax, label="PD reward")
    ax.fill_between(ep, (df.spd_r_mean-df.spd_r_std), (df.spd_r_mean+df.spd_r_std), color='b', alpha=.1)
    ax.fill_between(ep, (df.pd_r_mean-df.pd_r_std), (df.pd_r_mean+df.pd_r_std), color='g', alpha=.1)

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
    spd_experiment_dir = Path("/home/ji/experiments/walker_spd/mp_run_98d9_82fa")
    pd_experiment_dir = Path("/home/ji/experiments/walker_pd/mp_run_e125_9c2c")
    run_configs = list(spd_experiment_dir.rglob("run_config.yaml")) + list(pd_experiment_dir.rglob("run_config.yaml"))
    results_dir = Path("/home/ji/experiments/").joinpath("analysis")

    # fig, ax = plt.subplots()
    episodes = np.arange(100) + 1
    spd_updates = [0 for _ in range(100)]
    pd_updates = [0 for _ in range(100)]
    spd_reward = [[] for _ in range(100)]
    pd_reward = [[] for _ in range(100)]
    teacher_reward = []

    for run_config in run_configs:
        config = ParallelConfig.load(run_config)
        method_name = config.model_config.name
        try:
            results = CustomReport(
                config=config,
                parse_informative=False,
            ).results

            # print(results.columns)
            # print(len(results))

            for i, row in results.iterrows():
                train_ep = int(row['train_aux_traj_names'])
                updates = eval(method_name + '_updates')
                rewards = eval(method_name + '_reward')
                updates[train_ep] += int(row['num_updates'])
                rewards[train_ep].append(float(row['val_reward']))
                teacher_reward.append(float(row['train_reward']))

        except Exception as e:
            traceback.print_exc()
            raise e
        
    spd_reward = np.array(spd_reward)
    spd_updates = np.array(spd_updates) / 20
    pd_reward = np.array(pd_reward)
    pd_updates = np.array(pd_updates) / 20
    teacher_reward = np.mean(teacher_reward)

    plot_lines(results_dir, episodes, teacher_reward, spd_reward, spd_updates, pd_reward, pd_updates)