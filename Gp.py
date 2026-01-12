import streamlit as st
import random
import operator
import numpy as np
import pandas as pd

from deap import base, creator, gp, tools, algorithms

# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="GP Job Scheduling Optimization",
    layout="wide"
)

st.title("Multi-objective Job Scheduling using Genetic Programming")
st.write("Genetic Programming to evolve scheduling rules on limited machines")

# -------------------------------------------------
# SIDEBAR PARAMETERS
# -------------------------------------------------
st.sidebar.header("Simulation Parameters")

NUM_JOBS = st.sidebar.slider("Number of Jobs", 3, 10, 5)
NUM_MACHINES = st.sidebar.slider("Number of Machines", 1, 3, 2)
POP_SIZE = st.sidebar.slider("Population Size", 50, 300, 100)
GENS = st.sidebar.slider("Generations", 10, 100, 30)

random.seed(42)

# -------------------------------------------------
# JOB DATA
# -------------------------------------------------
jobs = []
for j in range(NUM_JOBS):
    jobs.append({
        "job_id": f"J{j+1}",
        "processing_time": random.randint(1, 10),
        "release_time": random.randint(0, 5)
    })

jobs_df = pd.DataFrame(jobs)

st.subheader("Job Data")
st.dataframe(jobs_df, use_container_width=True)

# -------------------------------------------------
# GP PRIMITIVES
# -------------------------------------------------
pset = gp.PrimitiveSet("MAIN", 2)
pset.renameArguments(ARG0="p_time", ARG1="wait_time")

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(min, 2)
pset.addPrimitive(max, 2)

pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))

# -------------------------------------------------
# FITNESS & INDIVIDUAL
# -------------------------------------------------
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# -------------------------------------------------
# SCHEDULING SIMULATION
# -------------------------------------------------
def simulate_schedule(rule_func):
    time = 0
    machine_available = [0] * NUM_MACHINES
    completion_times = []

    job_queue = jobs.copy()

    while job_queue:
        available_jobs = [j for j in job_queue if j["release_time"] <= time]

        if not available_jobs:
            time += 1
            continue

        scored_jobs = []
        for job in available_jobs:
            wait_time = time - job["release_time"]
            score = rule_func(job["processing_time"], wait_time)
            scored_jobs.append((score, job))

        scored_jobs.sort(key=lambda x: x[0])
        selected_job = scored_jobs[0][1]

        machine_index = machine_available.index(min(machine_available))
        start_time = max(time, machine_available[machine_index])
        finish_time = start_time + selected_job["processing_time"]

        machine_available[machine_index] = finish_time
        completion_times.append(finish_time)

        job_queue.remove(selected_job)
        time += 1

    makespan = max(completion_times)
    total_completion = sum(completion_times)
    idle_time = sum(machine_available) - makespan

    fitness = makespan + 0.1 * total_completion + 0.1 * idle_time
    return fitness

# -------------------------------------------------
# FITNESS FUNCTION
# -------------------------------------------------
def eval_individual(individual):
    func = toolbox.compile(expr=individual)
    return (simulate_schedule(func),)

toolbox.register("evaluate", eval_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=len, max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=10))

# -------------------------------------------------
# RUN GP
# -------------------------------------------------
if st.button("Run Genetic Programming"):
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=GENS,
        stats=stats,
        halloffame=hof,
        verbose=False
    )

    log_df = pd.DataFrame(logbook)

    st.subheader("Fitness Convergence")
    st.line_chart(
        log_df.set_index("gen")[["min"]],
        use_container_width=True
    )

    st.subheader("Best Evolved Scheduling Rule")
    st.code(str(hof[0]), language="text")

    st.subheader("Best Fitness Value")
    st.write(hof[0].fitness.values[0])
