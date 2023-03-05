import six
import sys
sys.modules['sklearn.externals.six'] = six

import pandas as pd
import mlrose_hiive as mlrose
import mlrose as mlrose_previous_version
import seaborn as sns
import matplotlib.pyplot as plt
import time

results = []
# problems_name = ["Flip Flop", "One Max", "Four Peaks"]
# fitness_functions = [mlrose.FlipFlop(), mlrose.OneMax(), mlrose.FourPeaks(t_pct=0.4)]
problems_name = ["Four Peaks"]
fitness_functions = [mlrose.FourPeaks(t_pct=0.4)]
problems = [mlrose.DiscreteOpt(length = 50, fitness_fn = fitness_function, maximize=True, max_val = 2) for fitness_function in fitness_functions]
for j in range(len(problems)):
    for i in range(1, 501, 100):
        #random_hill_climb
        start = time.time()
        fitness_score = mlrose.random_hill_climb(problems[j], max_attempts=100, max_iters=i, restarts=10, random_state=10)[1]
        results.append( [i, "random_hill_climb", problems_name[j], fitness_score, time.time() - start] )

        start = time.time()
        fitness_score = mlrose.simulated_annealing(problems[j], max_attempts=100, max_iters= i, random_state = 10)[1]
        results.append([i, "simulated_annealing", problems_name[j], fitness_score, time.time() - start])

        start = time.time()
        # fitness_score = mlrose.genetic_alg(problems[j], max_attempts=100, max_iters=i, pop_size=200, mutation_prob=0.1,random_state=10)[1]
        fitness_score = mlrose.genetic_alg(problems[j], max_attempts=100, max_iters=i, pop_size=100, mutation_prob=0.1,random_state=10)[1]

        results.append([i, "genetic_alg", problems_name[j], fitness_score, time.time() - start])

        start = time.time()
        # fitness_score = mlrose_previous_version.mimic(problems[j], pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=i, curve=False,
        #                  random_state=10, fast_mimic=True)[1]
        fitness_score = mlrose_previous_version.mimic(problems[j], pop_size=100, keep_pct=0.2, max_attempts=50, max_iters=i,
                                      curve=False,
                                      random_state=10, fast_mimic=True)[1]
        results.append([i, "mimic", problems_name[j], fitness_score, time.time() - start])
        print(i)

df = pd.DataFrame(results, columns=["Iteration", "Algorithm", "Problem Name", "Fitness", "Train Time"])
df.to_csv('Optimization report.csv')
df = pd.read_csv('Optimization report.csv')
for problem in problems_name:
  sns.lineplot(data=df[df['Problem Name']==problem], x="Iteration", y="Fitness", hue="Algorithm").set_title(problem+ ": Fitness vs Iterations")
  # plt.savefig("results/Optimization-" + problem + ": Fitness vs Iterations" + ".png")
  plt.savefig("results/adhoc_Optimization-" + problem + ": Fitness vs Iterations" + ".png")
  plt.clf()
  sns.lineplot(data=df[df['Problem Name']==problem], x="Iteration", y="Train Time", hue="Algorithm").set_title(problem+  ": Time vs Iterations")
  # plt.savefig("results/Optimization-" + problem + ": Time vs Iterations" + ".png")
  plt.savefig("results/Adhoc_Optimization-" + problem + ": Time vs Iterations" + ".png")
  plt.clf()



df.groupby(['Algorithm', 'Problem'])['Fitness'].max()
df.groupby(['Algorithm', 'Problem'])['Time'].max()
df[df['Problem']=='Flip Flop'].groupby(['Algorithm', 'Problem'])['Time'].mean()
df[df['Problem']=='One Max'].groupby(['Algorithm', 'Problem'])['Time'].mean()
df[df['Problem']=='Four Peaks'].groupby(['Algorithm', 'Problem'])['Time'].mean()
