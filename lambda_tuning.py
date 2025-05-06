import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv

from main import run_simulation

# --- Parameters ---
lambdas = [0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79]
agent_types = ["bandit", "pomdp"]
n_outer_runs = 50
n_students = 400
n_questions = 300
epsilon = 0.1

# --- Directory Setup ---
base_dir = "results"
os.makedirs(base_dir, exist_ok=True)
timestamp = datetime.now().strftime("lambda_tuning_%Y-%m-%d_%H-%M-%S")
results_dir = os.path.join(base_dir, timestamp)
os.makedirs(results_dir, exist_ok=True)
print(f"Saving lambda tuning results in: {results_dir}")

# --- Data Storage ---
learning_curves = {atype: {lam: [] for lam in lambdas} for atype in agent_types}
engagement_curves = {atype: {lam: [] for lam in lambdas} for atype in agent_types}
summary_data = []

# --- Main Loop ---
for lam in lambdas:
    print(f"\n=== Running λ = {lam} ===")
    for agent_type in agent_types:
        print(f"  Agent: {agent_type.upper()}")
        for run in range(n_outer_runs):
            results = run_simulation(
                n_students=n_students,
                n_questions=n_questions,
                reward_key="composite",
                epsilon=epsilon,
                lambda_param=lam,
                agent_type=agent_type
            )

            # Knowledge curve
            states_array = np.array([s + [np.nan]*(n_questions-len(s)) for s in results['states_by_question']])
            mean_states = np.nanmean(states_array, axis=0)
            learning_curves[agent_type][lam].append(mean_states)

            # Engagement curve
            engagement_array = np.array([s + [np.nan]*(n_questions-len(s)) for s in results['engagement_curve']])
            mean_engagement = np.nanmean(engagement_array, axis=0)
            engagement_curves[agent_type][lam].append(mean_engagement)

            # Summary
            summary_data.append([
                lam, agent_type, run + 1,
                np.mean(results['final_state']),
                np.mean(results['avg_state']),
                np.mean(results['accuracy']),
                np.mean(results['engagement']),
                np.mean(results['questions_answered']),
                np.mean(results['avg_total_reward'])
            ])

            print(f"    Run {run+1}/{n_outer_runs} done")

# --- Plot Curves ---
for agent_type in agent_types:
    # Knowledge
    plt.figure(figsize=(12, 6))
    for lam in lambdas:
        runs = np.array(learning_curves[agent_type][lam])
        avg = np.nanmean(runs, axis=0)
        std = np.nanstd(runs, axis=0)
        label = f"λ={lam:.2f}"
        plt.plot(avg, label=label)
        plt.fill_between(range(n_questions), avg - std, avg + std, alpha=0.2)
    plt.title(f"{agent_type.upper()} | Avg. Knowledge Over Time (mean ± std)")
    plt.xlabel("Question Number")
    plt.ylabel("Avg. Knowledge State")
    plt.legend()
    plt.tight_layout()
    fname = f"{agent_type}_knowledge.png"
    plt.savefig(os.path.join(results_dir, fname))
    plt.close()
    print(f"Saved: {fname}")

    # Engagement
    plt.figure(figsize=(12, 6))
    for lam in lambdas:
        runs = np.array(engagement_curves[agent_type][lam])
        avg = np.nanmean(runs, axis=0)
        std = np.nanstd(runs, axis=0)
        label = f"λ={lam:.1f}"
        plt.plot(avg, label=label)
        plt.fill_between(range(n_questions), avg - std, avg + std, alpha=0.2)
    plt.title(f"{agent_type.upper()} | Avg. Engagement Over Time (mean ± std)")
    plt.xlabel("Question Number")
    plt.ylabel("Avg. Engagement")
    plt.legend()
    plt.tight_layout()
    fname = f"{agent_type}_engagement.png"
    plt.savefig(os.path.join(results_dir, fname))
    plt.close()
    print(f"Saved: {fname}")

# --- CSV Summary ---
csv_path = os.path.join(results_dir, "summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "lambda", "agent_type", "run",
        "avg_final_state", "avg_avg_state", "avg_accuracy",
        "avg_engagement", "avg_questions_answered", "avg_total_reward"
    ])
    writer.writerows(summary_data)
print(f"Saved summary: {csv_path}")

# --- README ---
readme_path = os.path.join(results_dir, "README.txt")
with open(readme_path, "w") as f:
    f.write(
        f"Lambda Tuning Results\n"
        f"Timestamp: {timestamp}\n"
        f"Lambdas tested: {lambdas}\n"
        f"Agents: {agent_types}\n"
        f"Runs per lambda-agent pair: {n_outer_runs}\n"
        f"Students per run: {n_students}\n"
        f"Questions per student: {n_questions}\n"
        f"Epsilon (bandit): {epsilon}\n"
        f"Each graph shows mean ± std over runs\n"
    )
print(f"Saved README: {readme_path}")
