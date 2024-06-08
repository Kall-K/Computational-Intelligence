# One Max Problem
# https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
import random
from numpy.random import randint
from numpy.random import rand
import json
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NGRAM = (1,1) 
TOKEN_MODE = 'word'
MIN_DOCUMENT_FREQUENCY = 1 # all the words

top_k = {}

def load_top_k():
	with open('PartB/topk_texts.json', 'r', encoding='utf-8') as f:
		top_k = json.load(f)

	rows = [0]
	for rec in top_k:
		rows.append(rec['id'])

	df = pd.read_csv('PartB/region1693.csv', skiprows = lambda x: x not in rows)
	return df['text'].values.tolist()


def load_token(dec_num):
	with open('PartB/token_id.json', 'r', encoding='utf-8') as f:
		tokens = json.load(f)
	return tokens.get(str(dec_num))


def ngram_vectorize(x):
	# Create keyword arguments to pass to the 'tf-idf' vectorizer.
	kwargs = {
			'ngram_range': NGRAM,  # Use 1-grams.
			'strip_accents': 'unicode',
			'decode_error': 'ignore',
			'analyzer': TOKEN_MODE,  # Split text into word tokens.
			'min_df': MIN_DOCUMENT_FREQUENCY
	}

	vectorizer = TfidfVectorizer(**kwargs)
	return vectorizer.fit_transform(x)

# objective function
def cos_similarity(x):
	texts = load_top_k()
	text = load_token(bin2dec(x[:11])) + ' αλεξανδρε ουδις ' + load_token(bin2dec(x[11:]))
	texts.append(text)
	tfidf_matrix = ngram_vectorize(texts)
	specific_text_vector = tfidf_matrix[-1]
	cosine_similarities = cosine_similarity(tfidf_matrix[:-1], specific_text_vector)
	cosine_similarities = cosine_similarities.flatten()
	return sum(cosine_similarities)/len(cosine_similarities)


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

# selection
def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    relative_fitness = [f / total_fitness for f in fitness_scores]
    cumulative_probability = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]

    rand = random.random()
    for i, cp in enumerate(cumulative_probability):
        if rand <= cp:
            return population[i]

# crossover two parents to create two children
def crossover2(p1, p2, r_cross):#delete it
	c1, c2 = p1.copy(), p2.copy()
	if rand() < r_cross:
		mask = [random.randint(0, 1) for _ in range(len(p1))]
		c1 = [p1[i] if mask[i] == 1 else p2[i] for i in range(len(p1))]
		c2 = [p2[i] if mask[i] == 1 else p1[i] for i in range(len(p1))]
	return c1, c2

def crossover(p1, p2, r_cross):
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
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

def termination_check(elits, gens):
	rate = 1 
	c = 0
	if len(elits) > 1:
		rate = ((elits[-1]-elits[-2])/elits[-1])*100
	# print(rate)
	if len(elits) > gens:
		for i in range(gens):
			if (elits[-i-1] - elits[-i-2] < 0):
				# print("smaller")
				c += 1
			else: 
				break

	if (rate < 1 and rate != 0) or c == gens:
		return True
	else:
		return False
	
# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = []
	for _ in range(n_pop):
		new = randint(0, 2, n_bits).tolist()
		pop.append(check_range(new[:11])+check_range(new[11:]))

	# keep track of best solution
	best, best_eval = 0, objective(pop[0])
	elits = []
	# enumerate generations
	for gen in range(n_iter):
		# evaluate all candidates in the population
		scores = [objective(c) for c in pop]
		# find the elite of this gene
		elit_val = max(scores)
		elit_index = [i for i, s in enumerate(scores) if s == elit_val]
		# store the elit of this gene
		elits.append(elit_val)
		if termination_check(elits, gens=3):
			print('Terminated')
			break
		# check for new best solution
		for i in range(n_pop):
			len(scores)
			if scores[i] > best_eval:
				best, best_eval = pop[i], scores[i]
				# print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
		# select parents
		selected = [roulette_wheel_selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()		
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			elit_found = False
			if (i in elit_index):
				elit = 1
				elit_found = True
			elif (i+1 in elit_index):
				elit = 2
				elit_found = True
			# crossover and mutation
			iter = 0
			for c in crossover(p1, p2, r_cross):
				iter += 1
				# mutation
				if (elit_found == True and iter == elit):
					# store for next generation
					children.append(check_range(c[:11])+check_range(c[11:]))
				else:
					mutation(c, r_mut)
					# store for next generation
					children.append(check_range(c[:11])+check_range(c[11:]))
		# replace population
		pop = children
	return [best, best_eval]

# define the total iterations
n_iter = 1000
# bits
n_bits = 22
# define the population size
n_pop = 20
# crossover rate
r_cross = 0.1
# mutation rate
r_mut = 0.01#1.0 / float(n_bits)
# perform the genetic algorithm search
for _ in range(10):
	best, score = genetic_algorithm(cos_similarity, n_bits, n_iter, n_pop, r_cross, r_mut)
	print('Done!')
	print('f(%s) = %f' % (best, score))
	print(bin2dec(best[:11]), bin2dec(best[11:]))
	print('============================')
