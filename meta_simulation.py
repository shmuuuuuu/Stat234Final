import numpy as np
import matplotlib.pyplot as plt
from main import run_simulation, reward_keys, ORACLE_REWARD_FUNCTIONS, oracle_reward_keys
import os
from datetime import datetime
import csv

# --- Adjustable Parameters ---
n_outer_runs = 50
n_students = 400
n_questions = 300
lambda_param = 0.7
epsilon = 0.1
agent_types = ["bandit", "pomdp", "oracle"]

# --- Directory Setup ---
base_results_dir = "results"
os.makedirs(base_results_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
results_dir = os.path.join(base_results_dir, f"results_{timestamp}")
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved in: {results_dir}/")

# --- Data Storage ---
def agent_reward_keys(agent_type):
    if agent_type == "oracle":
        return oracle_reward_keys
    else:
        return reward_keys

learning_curves = {atype: {k: [] for k in agent_reward_keys(atype)} for atype in agent_types}
engagement_curves = {atype: {k: [] for k in agent_reward_keys(atype)} for atype in agent_types}
final_states = {atype: {k: [] for k in agent_reward_keys(atype)} for atype in agent_types}
avg_states = {atype: {k: [] for k in agent_reward_keys(atype)} for atype in agent_types}
accuracies = {atype: {k: [] for k in agent_reward_keys(atype)} for atype in agent_types}
avg_engagements = {atype: {k: [] for k in agent_reward_keys(atype)} for atype in agent_types}
questions_answered = {atype: {k: [] for k in agent_reward_keys(atype)} for atype in agent_types}
avg_total_rewards = {atype: {k: [] for k in agent_reward_keys(atype)} for atype in agent_types}

# --- Meta-Simulation Loop ---
for agent_type in agent_types:
    print(f"\n=== AGENT: {agent_type.upper()} ===")
    ar_keys = agent_reward_keys(agent_type)
    for idx, reward_key in enumerate(ar_keys):
        print(f"  Reward: {reward_key}")
        for run in range(n_outer_runs):
            if agent_type == "oracle":
                results = run_simulation(
                    n_students=n_students,
                    n_questions=n_questions,
                    reward_key=reward_key,
                    epsilon=epsilon,
                    lambda_param=lambda_param,
                    agent_type="oracle",
                    oracle_reward_fn=ORACLE_REWARD_FUNCTIONS[idx]
                )
            else:
                results = run_simulation(
                    n_students=n_students,
                    n_questions=n_questions,
                    reward_key=reward_key,
                    epsilon=epsilon,
                    lambda_param=lambda_param,
                    agent_type=agent_type
                )

            states_array = np.array([s + [np.nan]*(n_questions-len(s)) for s in results['states_by_question']])
            mean_states = np.nanmean(states_array, axis=0)
            learning_curves[agent_type][reward_key].append(mean_states)

            engagement_array = np.array([s + [np.nan]*(n_questions-len(s)) for s in results['engagement_curve']])
            mean_engagement = np.nanmean(engagement_array, axis=0)
            engagement_curves[agent_type][reward_key].append(mean_engagement)

            final_states[agent_type][reward_key].append(np.mean(results['final_state']))
            avg_states[agent_type][reward_key].append(np.mean(results['avg_state']))
            accuracies[agent_type][reward_key].append(np.mean(results['accuracy']))
            avg_engagements[agent_type][reward_key].append(np.mean(results['engagement']))
            questions_answered[agent_type][reward_key].append(np.mean(results['questions_answered']))
            avg_total_rewards[agent_type][reward_key].append(np.mean(results['avg_total_reward']))

            print(f"    Simulation {run+1}/{n_outer_runs} for reward '{reward_key}' completed.")

# --- Plot Knowledge Curves ---
for agent_type in agent_types:
    ar_keys = agent_reward_keys(agent_type)
    plt.figure(figsize=(12, 6))
    for reward_key in ar_keys:
        runs = np.array(learning_curves[agent_type][reward_key])
        avg = np.nanmean(runs, axis=0)
        std = np.nanstd(runs, axis=0)
        label = reward_key
        if reward_key == "composite":
            label += f" (λ={lambda_param})"
        plt.plot(avg, label=label)
        plt.fill_between(range(n_questions), avg-std, avg+std, alpha=0.2)
    plt.xlabel("Question Number")
    plt.ylabel("Avg. Knowledge State")
    plt.title(f"{agent_type.upper()} | Knowledge Over Time (λ={lambda_param}, ε={epsilon})")
    plt.legend()
    plt.tight_layout()
    fname = f"{agent_type}_knowledge.png"
    plt.savefig(os.path.join(results_dir, fname))
    plt.close()
    print(f"Saved knowledge plot: {fname}")

# --- Plot Engagement Curves ---
for agent_type in agent_types:
    ar_keys = agent_reward_keys(agent_type)
    plt.figure(figsize=(12, 6))
    for reward_key in ar_keys:
        runs = np.array(engagement_curves[agent_type][reward_key])
        avg = np.nanmean(runs, axis=0)
        std = np.nanstd(runs, axis=0)
        label = reward_key
        if reward_key == "composite":
            label += f" (λ={lambda_param})"
        plt.plot(avg, label=label)
        plt.fill_between(range(n_questions), avg-std, avg+std, alpha=0.2)
    plt.xlabel("Question Number")
    plt.ylabel("Avg. Engagement")
    plt.title(f"{agent_type.upper()} | Engagement Over Time (λ={lambda_param}, ε={epsilon})")
    plt.legend()
    plt.tight_layout()
    fname = f"{agent_type}_engagement.png"
    plt.savefig(os.path.join(results_dir, fname))
    plt.close()
    print(f"Saved engagement plot: {fname}")

# --- Save CSV ---
summary_csv_path = os.path.join(results_dir, "summary.csv")
header = [
    "agent_type", "reward_function", "run_number",
    "avg_final_state", "avg_avg_state", "avg_accuracy",
    "avg_engagement", "avg_questions_answered", "avg_total_reward"
]
summary_rows = []
for agent_type in agent_types:
    ar_keys = agent_reward_keys(agent_type)
    for idx, reward_key in enumerate(ar_keys):
        fs = final_states[agent_type][reward_key]
        avgs = avg_states[agent_type][reward_key]
        accs = accuracies[agent_type][reward_key]
        engs = avg_engagements[agent_type][reward_key]
        qs = questions_answered[agent_type][reward_key]
        avg_rwds = avg_total_rewards[agent_type][reward_key]
        for run in range(n_outer_runs):
            summary_rows.append([
                agent_type, reward_key, run+1,
                fs[run], avgs[run], accs[run], engs[run], qs[run], avg_rwds[run]
            ])
with open(summary_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(summary_rows)
print(f"Saved summary: {summary_csv_path}")

# --- Save README ---
readme_path = os.path.join(results_dir, "README.txt")
with open(readme_path, "w") as f:
    f.write(
        f"Meta-simulation results for RL in Education\n"
        f"Date: {timestamp}\n"
        f"Agent types: {agent_types}\n"
        f"Reward functions: {reward_keys}\n"
        f"Oracle reward functions: {oracle_reward_keys}\n"
        f"Meta-runs: {n_outer_runs}\n"
        f"Students per run: {n_students}\n"
        f"Questions per student: {n_questions}\n"
        f"Lambda (composite weight): {lambda_param}\n"
        f"Epsilon (bandit exploration): {epsilon}\n"
        f"Each plot shows mean ± std across meta-runs.\n"
    )
print(f"Saved README: {readme_path}")

# --- Console Summary ---
print("\nSummary of Results (mean ± std):")
for agent_type in agent_types:
    ar_keys = agent_reward_keys(agent_type)
    print(f"\n=== AGENT: {agent_type.upper()} ===")
    for reward_key in ar_keys:
        fs = np.array(final_states[agent_type][reward_key])
        qs = np.array(questions_answered[agent_type][reward_key])
        avg_rwds = np.array(avg_total_rewards[agent_type][reward_key])
        label = reward_key
        if reward_key == "composite":
            label += f" (λ={lambda_param})"
        print(f"{label:25}: Final state = {np.mean(fs):.2f} ± {np.std(fs):.2f} | "
              f"Questions = {np.mean(qs):.1f} ± {np.std(qs):.1f} | "
              f"Total reward = {np.mean(avg_rwds):.2f} ± {np.std(avg_rwds):.2f}")
