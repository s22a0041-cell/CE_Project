import streamlit as st
import random
import numpy as np
import pandas as pd

# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="GP Job Scheduling Optimization",
    layout="wide"
)

st.title("Job Scheduling using Genetic Programming")
st.write("Genetic Programming to evolve scheduling rules using uploaded CSV data")

# -------------------------------------------------
# SIDEBAR PARAMETERS
# -------------------------------------------------
st.sidebar.header("Simulation Parameters")

NUM_MACHINES = st.sidebar.slider("Number of Machines", 1, 3, 2)
POP_SIZE = st.sidebar.slider("Population Size", 20, 200, 50)
GENERATIONS = st.sidebar.slider("Generations", 10, 100, 30)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.2)

# -------------------------------------------------
# CSV UPLOAD
# -------------------------------------------------
st.subheader("Upload Job Dataset (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to start the simulation.")
    st.stop()

jobs_df = pd.read_csv(uploaded_file)

required_cols = {"job_id", "processing_time", "release_time"}
if not required_cols.issubset(jobs_df.columns):
    st.error("CSV must contain columns: job_id, processing_time, release_time")
    st.stop()

st.subheader("Uploaded Job Data")
st.dataframe(jobs_df, use_container_width=True)

jobs = jobs_df.to_dict(orient="records")

# -------------------------------------------------
# GP REPRESENTATION (RULE AS EXPRESSION STRING)
# p = processing_time, w = waiting_time
# -------------------------------------------------
OPERATORS = ["+", "-", "*"]
TERMINALS = ["p", "w", "1", "2"]

def random_expression(depth=2):
    if depth == 0:
        return random.choice(TERMINALS)
    return f"({random_expression(depth-1)} {random.choice(OPERATORS)} {random_expression(depth-1)})"

def evaluate_expression(expr, p, w):
    try:
        return eval(expr, {"p": p, "w": w})
    except:
        return float("inf")

# -------------------------------------------------
# SCHEDULING SIMULATION
# -------------------------------------------------
def simulate(rule_expr):
    time = 0
    machines = [0] * NUM_MACHINES
    completion_times = []
    job_queue = jobs.copy()

    while job_queue:
        available_jobs = [j for j in job_queue if j["release_time"] <= time]

        if not available_jobs:
            time += 1
            continue

        scores = []
        for job in available_jobs:
            waiting = time - job["release_time"]
            score = evaluate_expression(
                rule_expr,
                job["processing_time"],
                waiting
            )
            scores.append((score, job))

        scores.sort(key=lambda x: x[0])
        selected_job = scores[0][1]

        m_idx = machines.index(min(machines))
        start_time = max(time, machines[m_idx])
        finish_time = start_time + selected_job["processing_time"]

        machines[m_idx] = finish_time
        completion_times.append(finish_time)

        job_queue.remove(selected_job)
        time += 1

    makespan = max(completion_times)
    total_completion = sum(completion_times)
    idle_time = sum(machines) - makespan

    fitness = makespan + 0.1 * total_completion + 0.1 * idle_time
    return fitness

# -------------------------------------------------
# GENETIC OPERATORS
# -------------------------------------------------
def crossover(parent1, parent2):
    cut1 = random.randint(1, len(parent1) - 2)
    cut2 = random.randint(1, len(parent2) - 2)
    return parent1[:cut1] + parent2[cut2:]

def mutate(expr):
    if random.random() < MUTATION_RATE:
        return random_expression(2)
    return expr

def tournament_selection(population, fitness, k=3):
    selected = random.sample(list(zip(population, fitness)), k)
    selected.sort(key=lambda x: x[1])
    return selected[0][0]

# -------------------------------------------------
# RUN GP
# -------------------------------------------------
if st.button("Run Genetic Programming"):
    population = [random_expression(2) for _ in range(POP_SIZE)]
    best_fitness_history = []
    best_rule = None

    for gen in range(GENERATIONS):
        fitness_values = [simulate(ind) for ind in population]
        best_idx = int(np.argmin(fitness_values))

        best_fitness_history.append(fitness_values[best_idx])
        best_rule = population[best_idx]

        new_population = []
        for _ in range(POP_SIZE):
            p1 = tournament_selection(population, fitness_values)
            p2 = tournament_selection(population, fitness_values)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    st.subheader("Fitness Convergence")
    st.line_chart(
        pd.DataFrame({"Best Fitness": best_fitness_history}),
        use_container_width=True
    )

    st.subheader("Best Evolved Scheduling Rule")
    st.code(best_rule, language="text")

    st.subheader("Best Fitness Value")
    st.write(best_fitness_history[-1])
