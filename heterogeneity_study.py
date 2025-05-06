import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime

import main
from main import run_simulation, reward_keys

# --- Step 1: Define Specialized Student Types ---

class FastLearner(main.SimulatedStudent):
    def __init__(self, n_questions=200, n_states=5):
        super().__init__(n_questions, n_states)
        self.alpha = 0.25
        self.beta = 0.05
        self.motivation = 0.1
        self.engagement_drop = 0.8

class SteadyLearner(main.SimulatedStudent):
    def __init__(self, n_questions=200, n_states=5):
        super().__init__(n_questions, n_states)
        self.alpha = 0.05
        self.beta = 0.05
        self.motivation = 0.2
        self.engagement_drop = 0.6

class ForgetfulLearner(main.SimulatedStudent):
    def __init__(self, n_questions=200, n_states=5):
        super().__init__(n_questions, n_states)
        self.alpha = 0.15
        self.beta = 0.15
        self.motivation = -0.1
        self.engagement_drop = 1.1

class RandomLearner(main.SimulatedStudent):
    def __init__(self, n_questions=200, n_states=5):
        super().__init__(n_questions, n_states)

# --- Meta Simulation Settings ---
n_outer_runs = 50
n_students = 400
n_questions = 300
agent_types = ["bandit", "pomdp"]

student_classes = {
    "FAST": FastLearner,
    "STEADY": SteadyLearner,
    "FORGETFUL": ForgetfulLearner,
    "RANDOM": RandomLearner
}

# --- Output Setup ---
base_dir = "results"
timestamp = datetime.now().strftime("heterogeneity_full_%Y-%m-%d_%H-%M-%S")
results_dir = os.path.join(base_dir, timestamp)
os.makedirs(results_dir, exist_ok=True)
print(f"Saving heterogeneity results to: {results_dir}")

# --- Results Containers ---
summary_data = []

for reward_key in reward_keys:
    learning_curves = {atype: {stype: [] for stype in student_classes} for atype in agent_types}
    engagement_curves = {atype: {stype: [] for stype in student_classes} for atype in agent_types}

    for student_type, student_cls in student_classes.items():
        print(f"\\n=== STUDENT TYPE: {student_type} | REWARD: {reward_key} ===")

        original_student = main.SimulatedStudent
        main.SimulatedStudent = student_cls

        for agent_type in agent_types:
            print(f"  Agent: {agent_type.upper()}")
            for run in range(n_outer_runs):
                result = run_simulation(
                    n_students=n_students,
                    n_questions=n_questions,
                    reward_key=reward_key,
                    epsilon=0.1,
                    agent_type=agent_type
                )

                states_array = np.array([s + [np.nan]*(n_questions-len(s)) for s in result['states_by_question']])
                mean_states = np.nanmean(states_array, axis=0)
                learning_curves[agent_type][student_type].append(mean_states)

                engagement_array = np.array([s + [np.nan]*(n_questions-len(s)) for s in result['engagement_curve']])
                mean_engagement = np.nanmean(engagement_array, axis=0)
                engagement_curves[agent_type][student_type].append(mean_engagement)

                summary_data.append([
                    reward_key, student_type, agent_type, run + 1,
                    np.mean(result['final_state']),
                    np.mean(result['avg_state']),
                    np.mean(result['accuracy']),
                    np.mean(result['engagement']),
                    np.mean(result['questions_answered']),
                    np.mean(result['avg_total_reward'])
                ])
                print(f"    Run {run+1}/{n_outer_runs} complete")

        main.SimulatedStudent = original_student

    # --- Plotting ---
    for agent_type in agent_types:
        # Knowledge
        plt.figure(figsize=(12, 6))
        for student_type in student_classes:
            runs = np.array(learning_curves[agent_type][student_type])
            avg = np.nanmean(runs, axis=0)
            std = np.nanstd(runs, axis=0)
            plt.plot(avg, label=student_type)
            plt.fill_between(range(n_questions), avg - std, avg + std, alpha=0.2)
        plt.title(f"{agent_type.upper()} | Knowledge | Reward: {reward_key}")
        plt.xlabel("Question Number")
        plt.ylabel("Avg. Knowledge State")
        plt.legend()
        plt.tight_layout()
        fname = f"{agent_type}_knowledge_{reward_key}.png"
        plt.savefig(os.path.join(results_dir, fname))
        plt.close()

        # Engagement
        plt.figure(figsize=(12, 6))
        for student_type in student_classes:
            runs = np.array(engagement_curves[agent_type][student_type])
            avg = np.nanmean(runs, axis=0)
            std = np.nanstd(runs, axis=0)
            plt.plot(avg, label=student_type)
            plt.fill_between(range(n_questions), avg - std, avg + std, alpha=0.2)
        plt.title(f"{agent_type.upper()} | Engagement | Reward: {reward_key}")
        plt.xlabel("Question Number")
        plt.ylabel("Avg. Engagement")
        plt.legend()
        plt.tight_layout()
        fname = f"{agent_type}_engagement_{reward_key}.png"
        plt.savefig(os.path.join(results_dir, fname))
        plt.close()

# --- Save Summary CSV ---
summary_csv_path = os.path.join(results_dir, "summary.csv")
with open(summary_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "reward_function", "student_type", "agent_type", "run",
        "final_state", "avg_state", "accuracy",
        "engagement", "questions_answered", "avg_total_reward"
    ])
    writer.writerows(summary_data)

# --- README ---
readme_path = os.path.join(results_dir, "README.txt")
with open(readme_path, "w") as f:
    f.write(
        f"Heterogeneity Study (All Reward Functions)\\n"
        f"Timestamp: {timestamp}\\n"
        f"Student Types: {list(student_classes.keys())}\\n"
        f"Agents: {agent_types}\\n"
        f"Reward Functions: {reward_keys}\\n"
        f"Runs per type: {n_outer_runs}\\n"
        f"Students per run: {n_students}\\n"
        f"Questions per student: {n_questions}\\n"
        f"Each plot shows mean ± std across runs.\\n"
    )
print(f"Saved README and summary CSV to {results_dir}")