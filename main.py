import numpy as np

# --- BLOCK 1: Simulated Student ---

class SimulatedStudent:
    def __init__(self, n_questions=200, n_states=5):
        self.n_states = n_states
        self.state = np.random.choice(range(n_states))
        self.alpha = np.random.uniform(0.05, 0.25)
        self.beta = np.random.uniform(0.05, 0.15)
        self.engagement_drop = np.random.uniform(0.6, 1.1)
        self.motivation = np.random.normal(0, 0.15)
        self.n_questions = n_questions
        self.t = 0
        self.frustration = 0
        self.frustration_threshold = 5
        self.wrong_streak = 0

        self.state_history = []
        self.correct_history = []
        self.engagement_history = []
        self.response_time_history = []
        self.difficulty_history = []
        self.state_gain_history = []
        self.quit_early_history = []
        self.time_allowance_history = []

    def answer_question(self, difficulty):
        mismatch = abs(self.state - difficulty)
        base_prob = [0.85, 0.5, 0.2, 0.05]
        base_prob = base_prob[mismatch] if mismatch < len(base_prob) else 0.01

        prob_correct = np.clip(base_prob + self.motivation, 0, 1)
        correct = int(np.random.rand() < prob_correct)

        preferred = max(self.state - 1, 0)
        engagement = 1.0 - self.engagement_drop * abs(preferred - difficulty) + self.motivation
        engagement += np.random.normal(0, 0.05)
        engagement = np.clip(engagement, 0, 1)

        base = 3
        a = 1.3
        b = 1.0
        tau = 0.7
        expected_time = base + a * difficulty + b * mismatch - 0.5 * self.motivation
        response_time = max(0, np.random.normal(expected_time, tau))
        time_allowance = base + a * difficulty

        if engagement < 0.2:
            self.frustration += 1
        else:
            self.frustration = max(0, self.frustration - 1)
        if correct:
            self.wrong_streak = 0
        else:
            self.wrong_streak += 1
        quit_early = (self.frustration >= self.frustration_threshold) or (self.wrong_streak >= 5)

        old_state = self.state
        if correct and (difficulty >= self.state):
            if np.random.rand() < self.alpha:
                self.state = min(self.state + 1, self.n_states - 1)
        elif not correct:
            if np.random.rand() < self.beta:
                self.state = max(self.state - 1, 0)
        state_gain = int(self.state > old_state)

        self.state_history.append(self.state)
        self.correct_history.append(correct)
        self.engagement_history.append(engagement)
        self.response_time_history.append(response_time)
        self.difficulty_history.append(difficulty)
        self.state_gain_history.append(state_gain)
        self.quit_early_history.append(quit_early)
        self.time_allowance_history.append(time_allowance)
        self.t += 1

        return {
            'state': self.state,
            'correct': correct,
            'engagement': engagement,
            'response_time': response_time,
            'difficulty': difficulty,
            'state_gain': state_gain,
            'quit_early': quit_early,
            'time_allowance': time_allowance
        }

    def reset(self):
        self.__init__(n_questions=self.n_questions, n_states=self.n_states)

# --- BLOCK 2: Reward Functions ---

def reward_correctness(obs, **kwargs):
    return obs['correct']

def reward_engagement(obs, **kwargs):
    return obs['engagement']

def make_composite_reward(lambda_param):
    def reward_composite(obs, **kwargs):
        return lambda_param * obs['correct'] + (1 - lambda_param) * obs['engagement']
    return reward_composite

def reward_knowledge_gain(obs, **kwargs):
    return obs['correct'] + obs['state_gain']

def reward_learning_progress(obs, avg_correct_prev, avg_correct_curr, **kwargs):
    return obs['correct'] + 2 * (avg_correct_curr - avg_correct_prev)

def reward_composite_time(obs, **kwargs):
    penalty = max(0, obs['response_time'] - obs['time_allowance'])
    return 0.6 * obs['correct'] + 0.3 * obs['engagement'] - 0.1 * penalty

REWARD_FUNCTIONS = {
    "correctness": reward_correctness,
    "engagement": reward_engagement,
    "knowledge_gain": reward_knowledge_gain,
    "learning_progress": reward_learning_progress,
    "composite_time": reward_composite_time
}

reward_keys = list(REWARD_FUNCTIONS.keys()) + ["composite"]

# --- BLOCK 3: Bandit Agent ---

class BanditAgent:
    def __init__(self, n_actions=5, epsilon=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.Q = np.zeros(n_actions)
        self.N = np.zeros(n_actions)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        return np.argmax(self.Q)

    def update(self, action, reward):
        self.N[action] += 1
        lr = 1 / self.N[action]
        self.Q[action] += lr * (reward - self.Q[action])

# --- BLOCK 4: POMDP Agent ---

class POMDPAgent:
    def __init__(self, n_states=5, n_actions=5, reward_fn=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.reward_fn = reward_fn
        self.belief = np.ones(n_states) / n_states
        self.p_correct = {0:0.85, 1:0.5, 2:0.2, 3:0.05, 4:0.01}

    def select_action(self):
        expected_rewards = []
        for a in range(self.n_actions):
            er = 0
            for s in range(self.n_states):
                mismatch = abs(s - a)
                base_prob = self.p_correct.get(mismatch, 0.01)
                obs = {
                    'state': s,
                    'correct': base_prob,
                    'engagement': self._expected_engagement(s, a),
                    'response_time': 5,
                    'time_allowance': 5,
                    'state_gain': 0
                }
                er += self.belief[s] * self.reward_fn(obs, avg_correct_prev=0, avg_correct_curr=0)
            expected_rewards.append(er)
        return np.argmax(expected_rewards)

    def update(self, action, obs):
        new_belief = np.zeros(self.n_states)
        for s in range(self.n_states):
            mismatch = abs(s - action)
            p_corr = self.p_correct.get(mismatch, 0.01)
            likelihood = p_corr if obs['correct'] else (1 - p_corr)
            new_belief[s] = likelihood * self.belief[s]
        self.belief = new_belief / np.sum(new_belief) if np.sum(new_belief) > 0 else np.ones(self.n_states) / self.n_states

    def _expected_engagement(self, state, difficulty):
        preferred = max(state - 1, 0)
        engagement_drop = 0.85
        motivation = 0
        engagement = 1.0 - engagement_drop * abs(preferred - difficulty) + motivation
        return np.clip(engagement, 0, 1)

# --- BLOCK 5: Oracle Agent + Reward Functions ---

class OracleAgent:
    def __init__(self, n_difficulties=5, oracle_reward_fn=None):
        self.n_difficulties = n_difficulties
        self.oracle_reward_fn = oracle_reward_fn

    def select_action(self, true_state=None, true_student=None, prev_corrects=None):
        best_d = 0
        best_val = -float('inf')
        kwargs = {}
        if self.oracle_reward_fn.__name__ == "oracle_reward_learning_progress":
            if prev_corrects and len(prev_corrects) > 0:
                window = 5
                t = len(prev_corrects)
                avg_correct_prev = np.mean(prev_corrects[max(0, t - window):t]) if t > 0 else 0
                avg_correct_curr = np.mean(prev_corrects[max(0, t - window + 1):t + 1])
            else:
                avg_correct_prev = avg_correct_curr = 0
            kwargs.update({"avg_correct_prev": avg_correct_prev, "avg_correct_curr": avg_correct_curr})

        for d in range(self.n_difficulties):
            val = self.oracle_reward_fn(true_student=true_student, true_state=true_state, d=d, **kwargs)
            if val > best_val:
                best_val = val
                best_d = d
        return best_d

    def update(self, *args, **kwargs):
        pass

def oracle_reward_correctness(true_student, true_state, d, **kwargs):
    mismatch = abs(true_state - d)
    base_prob = [0.85, 0.5, 0.2, 0.05]
    base_prob = base_prob[mismatch] if mismatch < len(base_prob) else 0.01
    prob = np.clip(base_prob + true_student.motivation, 0, 1)
    return prob

def oracle_reward_engagement(true_student, true_state, d, **kwargs):
    preferred = max(true_state - 1, 0)
    eta = true_student.engagement_drop
    m = true_student.motivation
    engagement = 1.0 - eta * abs(preferred - d) + m
    return np.clip(engagement, 0, 1)

def oracle_reward_composite(true_student, true_state, d, **kwargs):
    return 0.7 * oracle_reward_correctness(true_student, true_state, d) + 0.3 * oracle_reward_engagement(true_student, true_state, d)

def oracle_reward_knowledge_gain(true_student, true_state, d, **kwargs):
    return 1.0 if d >= true_state and true_state < true_student.n_states - 1 else 0.0

def oracle_reward_learning_progress(true_student, true_state, d, avg_correct_prev=0, avg_correct_curr=0, **kwargs):
    correctness = oracle_reward_correctness(true_student, true_state, d)
    return correctness + 2 * (avg_correct_curr - avg_correct_prev)

def oracle_reward_composite_time(true_student, true_state, d, **kwargs):
    base = 3
    a = 1.3
    b = 1.0
    m = true_student.motivation
    mismatch = abs(true_state - d)
    expected_time = base + a * d + b * mismatch - 0.5 * m
    time_allowance = base + a * d
    penalty = max(0, expected_time - time_allowance)
    correct = oracle_reward_correctness(true_student, true_state, d)
    engage = oracle_reward_engagement(true_student, true_state, d)
    return 0.6 * correct + 0.3 * engage - 0.1 * penalty

ORACLE_REWARD_FUNCTIONS = [
    oracle_reward_correctness,
    oracle_reward_engagement,
    oracle_reward_composite,
    oracle_reward_knowledge_gain,
    oracle_reward_learning_progress,
    oracle_reward_composite_time
]

oracle_reward_keys = [
    "correctness", "engagement", "composite", "knowledge_gain", "learning_progress", "composite_time"
]

def run_simulation(
    n_students=400,
    n_questions=200,
    reward_key="composite",
    epsilon=0.1,
    lambda_param=0.7,
    n_difficulties=5,
    agent_type="bandit",
    oracle_reward_fn=None,
    verbose=False
):
    results = {
        'total_reward': [],
        'final_state': [],
        'avg_state': [],
        'accuracy': [],
        'engagement': [],
        'avg_time': [],
        'within_time': [],
        'states_by_question': [],
        'engagement_curve': [],
        'questions_answered': [],
        'avg_total_reward': [],
    }

    # Choose reward function
    if reward_key == "composite":
        reward_fn = make_composite_reward(lambda_param)
    else:
        reward_fn = REWARD_FUNCTIONS.get(reward_key, None)

    for student_idx in range(n_students):
        student = SimulatedStudent(n_questions=n_questions, n_states=n_difficulties)

        # Choose agent
        if agent_type == "bandit":
            agent = BanditAgent(n_actions=n_difficulties, epsilon=epsilon)
        elif agent_type == "pomdp":
            agent = POMDPAgent(n_states=n_difficulties, n_actions=n_difficulties, reward_fn=reward_fn)
        elif agent_type == "oracle":
            agent = OracleAgent(n_difficulties=n_difficulties, oracle_reward_fn=oracle_reward_fn)
        else:
            raise ValueError("Unknown agent_type.")

        episode_reward = 0
        corrects = []
        engagements = []
        times = []
        within_time = []
        states_by_question = []
        engagement_curve = []
        answered_questions = 0
        total_rewards_this_student = []
        correct_hist = []
        window = 5

        for t in range(n_questions):
            if agent_type == "oracle":
                action = agent.select_action(
                    true_state=student.state,
                    true_student=student,
                    prev_corrects=correct_hist
                )
            else:
                action = agent.select_action()

            obs = student.answer_question(action)
            answered_questions += 1
            correct_hist.append(obs['correct'])

            avg_correct_prev = np.mean(correct_hist[max(0, t - window):t]) if t > 0 else 0
            avg_correct_curr = np.mean(correct_hist[max(0, t - window + 1):t + 1])

            # Get reward
            if agent_type == "oracle":
                if oracle_reward_fn.__name__ == "oracle_reward_learning_progress":
                    reward = oracle_reward_fn(
                        true_student=student,
                        true_state=student.state,
                        d=action,
                        avg_correct_prev=avg_correct_prev,
                        avg_correct_curr=avg_correct_curr
                    )
                else:
                    reward = oracle_reward_fn(true_student=student, true_state=student.state, d=action)
            elif reward_key == "learning_progress":
                reward = reward_fn(obs, avg_correct_prev=avg_correct_prev, avg_correct_curr=avg_correct_curr)
            else:
                reward = reward_fn(obs)

            if agent_type == "bandit":
                agent.update(action, reward)
            elif agent_type == "pomdp":
                agent.update(action, obs)

            episode_reward += reward
            total_rewards_this_student.append(reward)
            corrects.append(obs['correct'])
            engagements.append(obs['engagement'])
            times.append(obs['response_time'])
            within_time.append(obs['response_time'] <= obs['time_allowance'])
            states_by_question.append(obs['state'])
            engagement_curve.append(obs['engagement'])

            if obs['quit_early']:
                break

        results['total_reward'].append(episode_reward)
        results['final_state'].append(student.state_history[-1])
        results['avg_state'].append(np.mean(student.state_history))
        results['accuracy'].append(np.mean(corrects))
        results['engagement'].append(np.mean(engagements))
        results['avg_time'].append(np.mean(times))
        results['within_time'].append(np.mean(within_time))
        results['states_by_question'].append(states_by_question)
        results['engagement_curve'].append(engagement_curve)
        results['questions_answered'].append(answered_questions)
        results['avg_total_reward'].append(np.mean(total_rewards_this_student))

        if verbose and student_idx < 2:
            print(f"Student {student_idx} | Final state: {student.state_history[-1]}, Avg state: {np.mean(student.state_history):.2f}, Accuracy: {np.mean(corrects):.2f}")

    return results
