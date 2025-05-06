import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv
from main import run_simulation, reward_keys

# --- Configurable Parameters ---
epsilons = [round(e, 2) for e in np.arange(0.05, 0.45, 0.05)]
reward_functions = reward_keys
n_outer_runs = 50
n_students = 400
n_questions = 300

# --- Directory Setup ---
base_dir = "results"
os.makedirs(base_dir, exist_ok=True)
timestamp = datetime.now().strftime("epsilon_tuning_%Y-%m-%d_%H-%M-%S")
results_dir = os.path.join(base_dir, timestamp)
os.makedirs(results_dir, exist_ok=True)
print(f"Saving epsilon tuning results in: {results_dir}")

# --- Data Structures ---
learning_curves = {rk: {eps: [] for eps in epsilons} for rk in reward_functions}
engagement_curves = {rk: {eps: [] for eps in epsilons} for rk in reward_functions}
summary_data = []

# --- Main Loop ---
for reward_key in reward_functions:
    print(f"\n=== Reward Function: {reward_key} ===")
    for eps in epsilons:
        print(f"  Epsilon: {eps}")
        for run in range(n_outer_runs):
            results = run_simulation(
                n_students=n_students,
                n_questions=n_questions,
                reward_key=reward_key,
                epsilon=eps,
                agent_type="bandit"
            )

            # Knowledge
            states_array = np.array([s + [np.nan]*(n_questions-len(s)) for s in results['states_by_question']])
            mean_states = np.nanmean(states_array, axis=0)
            learning_curves[reward_key][eps].append(mean_states)

            # Engagement
            engagement_array = np.array([s + [np.nan]*(n_questions-len(s)) for s in results['engagement_curve']])
            mean_engagement = np.nanmean(engagement_array, axis=0)
            engagement_curves[reward_key][eps].append(mean_engagement)

            # Summary metrics
            summary_data.append([
                reward_key, eps, run + 1,
                np.mean(results['final_state']),
                np.mean(results['avg_state']),
                np.mean(results['accuracy']),
                np.mean(results['engagement']),
                np.mean(results['questions_answered']),
                np.mean(results['avg_total_reward'])
            ])
            print(f"    Run {run+1}/{n_outer_runs} complete")

# --- Plotting ---
for reward_key in reward_functions:
    # Knowledge
    plt.figure(figsize=(12, 6))
    for eps in epsilons:
        runs = np.array(learning_curves[reward_key][eps])
        avg = np.nanmean(runs, axis=0)
        std = np.nanstd(runs, axis=0)
        label = f"ε={eps:.2f}"
        plt.plot(avg, label=label)
        plt.fill_between(range(n_questions), avg - std, avg + std, alpha=0.2)
    plt.title(f"Bandit | Knowledge Over Time | Reward: {reward_key}")
    plt.xlabel("Question Number")
    plt.ylabel("Avg. Knowledge State")
    plt.legend()
    plt.tight_layout()
    fname = f"{reward_key}_knowledge.png"
    plt.savefig(os.path.join(results_dir, fname))
    plt.close()
    print(f"Saved: {fname}")

    # Engagement
    plt.figure(figsize=(12, 6))
    for eps in epsilons:
        runs = np.array(engagement_curves[reward_key][eps])
        avg = np.nanmean(runs, axis=0)
        std = np.nanstd(runs, axis=0)
        label = f"ε={eps:.2f}"
        plt.plot(avg, label=label)
        plt.fill_between(range(n_questions), avg - std, avg + std, alpha=0.2)
    plt.title(f"Bandit | Engagement Over Time | Reward: {reward_key}")
    plt.xlabel("Question Number")
    plt.ylabel("Avg. Engagement")
    plt.legend()
    plt.tight_layout()
    fname = f"{reward_key}_engagement.png"
    plt.savefig(os.path.join(results_dir, fname))
    plt.close()
    print(f"Saved: {fname}")

# --- CSV Summary ---
csv_path = os.path.join(results_dir, "summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "reward_function", "epsilon", "run",
        "avg_final_state", "avg_avg_state", "avg_accuracy",
        "avg_engagement", "avg_questions_answered", "avg_total_reward"
    ])
    writer.writerows(summary_data)
print(f"Saved summary: {csv_path}")

# --- README ---
readme_path = os.path.join(results_dir, "README.txt")
with open(readme_path, "w") as f:
    f.write(
        f"Epsilon Tuning Results\n"
        f"Timestamp: {timestamp}\n"
        f"Reward functions: {reward_functions}\n"
        f"Epsilons tested: {epsilons}\n"
        f"Agent: Bandit only\n"
        f"Runs per setting: {n_outer_runs}\n"
        f"Students per run: {n_students}\n"
        f"Questions per student: {n_questions}\n"
        f"Each plot shows mean ± std across runs\n"
    )
print(f"Saved README: {readme_path}")
