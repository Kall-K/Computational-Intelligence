import random
from numpy.random import randint
from numpy.random import rand
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import csv
import os
import shutil

# create folder and if exists clear folder contents
figures_path = 'PartB/figures/'
results_path = 'PartB/results/'

def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

clear_folder(figures_path)
clear_folder(results_path)

# load texts and top k texts
df = pd.read_csv('PartB/region1693.csv', encoding='utf-8')
TEXTS = df['text'].values.tolist()

with open('PartB/topk_texts.json', 'r', encoding='utf-8') as f:
    TOP_K = json.load(f)

# auxiliary functions
def load_token(dec_num):
	with open('PartB/token_id.json', 'r', encoding='utf-8') as f:
		tokens = json.load(f)
	return tokens.get(str(dec_num))

def bin2dec(binary):
    dec = 0
    for b in range(len(binary)):
        dec += binary[-b-1]*pow(2,b)
    return dec

def dec2bin(decimal):
    bin_num = bin(decimal)[2:]
    bin_num = bin_num.zfill(11)
    return [int(b) for b in bin_num]

def check_range(num):
    num = bin2dec(num)
    if num not in range(0,1677):
        num = random_remapping()
    return dec2bin(num)

def random_remapping():
    num = randint(0,1677)
    if num in range(0,1677):
        return num
    else:
        random_remapping()

# genetic algorith class
class GeneticAlgorithm:
    def __init__(self, n_bits, n_iter, n_pop, r_cross, r_mut):
        self.n_pop = n_pop
        self.r_cross = r_cross
        self.r_mut = r_mut
        self.n_bits = n_bits
        self.n_iter = n_iter
        self.objective = self.cos_similarity
        self.stagnation_threshold = 50  # Number of generations with no significant improvement
        self.improvement_threshold = 0.01  # Improvement percentage threshold

    # objective function
    def cos_similarity(self, x):
        main_text = load_token(bin2dec(x[:11])) + ' αλεξανδρε ουδις ' + load_token(bin2dec(x[11:]))
        all_texts = TEXTS + [main_text] 
        tfidf_matrix = TfidfVectorizer().fit_transform(all_texts)
        
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)
        indexes = [i['id'] for i in TOP_K]
        cosine_sim = []
        for i in range(len(cosine_sim_matrix[-1])):
            if i in indexes:
                cosine_sim.append(cosine_sim_matrix[-1][i])
            
        return sum(cosine_sim)/len(cosine_sim)

    # selection
    def roulette_wheel_selection(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        relative_fitness = [f / total_fitness for f in fitness_scores]
        cumulative_probability = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]

        rand = random.random()
        for i, cp in enumerate(cumulative_probability):
            if rand <= cp:
                return population[i]

    # crossover two parents to create two children
    def crossover(self, p1, p2, r_cross):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if rand() < r_cross:
            # select crossover point that is not on the end of the string
            pt = randint(1, len(p1)-2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]

            pt2 = randint(1, len(p1)-2)
            while ():
                if pt2 != pt:
                    break
                else :
                    pt2 = randint(1, len(p1)-2)
            c1 = c1[:pt2] + c2[pt2:]
            c2 = c2[:pt2] + c1[pt2:]

        return [c1, c2]

    # mutation operator
    def mutation(self, bitstring, r_mut):
        for i in range(len(bitstring)):
            # check for a mutation
            if rand() < r_mut:
                # flip the bit
                bitstring[i] = 1 - bitstring[i]

    # genetic algorithm
    def genetic_algorithm(self):
        # initial population of random bitstring
        pop = []
        for _ in range(self.n_pop):
            new = randint(0, 2, self.n_bits).tolist()
            pop.append(check_range(new[:11])+check_range(new[11:]))

        # keep track of best solution
        best, best_eval = pop[0], self.objective(pop[0])

        # initialization of variables
        elits = []
        no_improvement_generations = 0

        # enumerate generations
        for gen in range(self.n_iter):
            # evaluate all candidates in the population
            scores = [self.objective(c) for c in pop]
            # find the elite of this gene
            elit_val = max(scores)
            elit_index = [i for i, s in enumerate(scores) if s == elit_val]
            # store the elit of this gene
            elits.append(elit_val)

            # Termination criteria
            if len(elits) > 1:
                if elit_val <= elits[-2]:
                    no_improvement_generations += 1
                    criteria1 = no_improvement_generations >= self.stagnation_threshold
                    if criteria1:
                        print(f"Terminated at generation {gen+1}: criterion 1 (stagnation={self.stagnation_threshold}) ")
                        break
                else:
                    no_improvement_generations = 0
                    criteria2 = ((elit_val - elits[-2])/elits[-2]) < self.improvement_threshold
                    if criteria2:
                        print(f"Terminated at generation {gen+1}: criterion 2 (improvement={self.improvement_threshold})")
                        break
                if gen == self.n_iter - 1:
                    print("Terminated due to reaching the maximum number of generations")

            # check for new best solution
            for i in range(self.n_pop):
                len(scores)
                if scores[i] > best_eval:
                    best, best_eval = pop[i], scores[i]
                    print(">%d, new best f(%s) = %.3f" % (gen+1,  pop[i], scores[i]))
            # select parents
            selected = [self.roulette_wheel_selection(pop, scores) for _ in range(self.n_pop)]
            # create the next generation
            children = list()		
            for i in range(0, self.n_pop, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i+1]

                mutation = True
                elit = 0
                if i in elit_index:
                    mutation = False
                    elit = 1
                if i+1 in elit_index:
                    mutation = False
                    elit = 2
                
                iter = 0
                for c in self.crossover(p1, p2, self.r_cross):
                    iter += 1
                    # mutation
                    if (mutation == False and iter == elit):
                        # store for next generation without mutation
                        children.append(check_range(c[:11])+check_range(c[11:]))
                    else:
                        self.mutation(c, self.r_mut)
                        # store for next generation
                        children.append(check_range(c[:11])+check_range(c[11:]))
                   
            # replace population
            pop = children
        return [best, best_eval, elits]

def run_ga(n_bits, n_iter, n_pop, r_cross, r_mut):
            ga = GeneticAlgorithm(n_bits, n_iter, n_pop, r_cross, r_mut)
            best, best_eval, elits = ga.genetic_algorithm()
            return best, best_eval, elits

def main():
    with open('PartB/experiments.json', 'r', encoding='utf-8') as f:
        exp = json.load(f)
    for num, e in enumerate(exp):
        # define the total iterations
        n_iter = 1000
        # bits
        n_bits = 22
        # define the population size
        n_pop = e['n_pop'] #[20, 200]
        # crossover rate
        r_cross = e['r_cross'] #[0.1, 0.6, 0.9]
        # mutation rate
        r_mut = e['r_mut'] #[0, 0.01, 0.1]            
        # perform the genetic algorithm search
        print(f'>Experiment {num+1}: n_pop={n_pop}, r_cross={r_cross}, rmut={r_mut}')
        print('>>>Start')

        fields = ['population size', 'crossover probability', 'mutation probability', 'best', 'generations', 'text']
        all_elits = np.zeros((10, 1000))
        avg_best_eval = 0
        avg_gen = 0
        with open(f'PartB/results/experiment{num+1}.csv', 'w', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            for i in range(10):
                best, best_eval, elits = run_ga(n_bits, n_iter, n_pop, r_cross, r_mut)
                avg_best_eval += best_eval
                avg_gen += len(elits)
                text = load_token(bin2dec(best[:11])) + ' αλεξανδρε ουδις ' + load_token(bin2dec(best[11:]))
                print('>Done!')
                print('f(%s) = %f' % (best, best_eval))
                print(f'word 1:{bin2dec(best[:11])}\tword 2:{bin2dec(best[11:])}')
                print(text)
                print('-----------------------------------------------------------------------------')
                for j in range(len(elits)):
                    all_elits[i][j] = elits[j]
                writer.writerow({'population size':n_pop,'crossover probability':r_cross,
                                  'mutation probability':r_mut, 'best':best_eval, 'generations':len(elits), 'text':text})
            writer.writerow({'population size':n_pop,'crossover probability':r_cross,
                              'mutation probability':r_mut, 'best':avg_best_eval/10, 'generations':avg_gen/10})
        column_averages = np.mean(all_elits, axis=0)
        x_values = np.arange(1, 1001)

        # Plotting the data
        plt.figure(figsize=(10, 5))
        plt.plot(x_values, column_averages, label='Generation Averages')
        plt.xlabel('Generation')
        plt.ylabel('Average Elite Value')
        plt.title(f'Average Elite Value of each Generation, Exp:{num+1}')
        plt.legend()
        # plt.show()
        plt.savefig(f'PartB/figures/experiment{num+1}.png')
        print('>>>End')

main()