#!/usr/bin/env python

"""
__author__ = "Lech Szymanski"
__copyright__ = "Copyright 2019, COSC343"
__license__ = "GPL"
__version__ = "2.0.1"
__maintainer__ = "Lech Szymanski"
__email__ = "lechszym@cs.otago.ac.nz"
"""

from cosc343world import Creature, World
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from scipy import stats

# You can change this number to specify how many generations creatures are going to evolve over...
numGenerations = 100

# You can change this number to specify how many turns in simulation of the world for given generation
numTurns = 100

# You can change this number to change the world type.  You have two choices - world 1 or 2 (described in
# the assignment 2 pdf document)
worldType = 2

# You can change this number to change the world size 24
gridSize = 48

# You can set this mode to True to have same initial conditions for each simulation in each generation.  Good
# for development, when you want to have some determinism in how the world runs from generatin to generation.
repeatableMode = False

# This is a class implementing you creature a.k.a MyCreature.  It extends the basic Creature, which provides the
# basic functionality of the creature for the world simulation.  Your job is to implement the AgentFunction
# that controls creature's behavoiur by producing actions in respons to percepts.

average_fitness = []

class MyCreature(Creature):

    # Initialisation function.  This is where you creature
    # should be initialised with a chromosome in random state.  You need to decide the format of your
    # chromosome and the model that it's going to give rise to
    #
    # Input: numPercepts - the size of percepts list that creature will receive in each turn
    #        numActions - the size of actions list that creature must create on each turn
    def __init__(self, numPercepts, numActions):

        # Place your initialisation code here.  Ideally this should set up the creature's chromosome
        # and set it to some random state.
        self.chromosome = np.random.uniform(0, 1, (numActions, numPercepts * 4))
        self.totalPercepts = numPercepts
        self.score = 0

        # Do not remove this line at the end.  It calls constructors
        # of the parent classes.
        Creature.__init__(self)

    # This is the implementation of the agent function that is called on every turn, giving your
    # creature a chance to perform an action.  You need to implement a model here, that takes its parameters
    # from the chromosome and it produces a set of actions from provided percepts
    #
    # Input: percepts - a list of percepts
    #        numAction - the size of the actions list that needs to be returned

    def AgentFunction(self, percepts, numActions):
        # At the moment the actions is a list of random numbers.  You need to
        # replace this with some model that maps percepts to actions.  The model
        # should be parametrised by the chromosome
        dummy_variables = np.full(4 * self.totalPercepts, 0)
        x = 0
        for p in percepts:
            dummy_variables[x + int(p)] = 1
            x += 4
        actions = np.sum((self.chromosome * dummy_variables), axis=1).tolist()

        return actions

def fitness_fn(turns, energy, isDead):
    if (isDead):
        return (3*turns) + energy
    else:
        return (3*turns) + energy + 120

def selection(population, tf, s):
    scores = []
    for p in population:
        scores.append(p.score/tf)
    return np.random.choice(population, len(population) - s, scores)

def tournament_selection(population,s):
    sample = []
    while (len(sample) < len(population) - s):
        sample.append(max(np.random.choice(population, size=4),key=lambda x: x.score))
    return sample

def recombine(x, y):
    s = random.randrange(0, 9)
    if np.random.choice([True, False]):
        j = x.chromosome[9:]
        k = y.chromosome[9:]
    else:
        j = y.chromosome[9:]
        k = x.chromosome[9:]
    return ([*x.chromosome[:s], *y.chromosome[s:9],*j], [*y.chromosome[:s], *x.chromosome[s:9], *k])


def mutate(c, gp, pmut):
    if(np.random.choice([True, False], p=[pmut, 1-pmut])):
        s = random.randrange(0, len(c))
        new_gene = gp[random.randrange(0, len(gp))]
        newChromosome = c[:s] + [new_gene] + c[s+1:]
        return newChromosome
    else:
        return c

# This function is called after every simulation, passing a list of the old population of creatures, whose fitness
# you need to evaluate and whose chromosomes you can use to create new creatures.
#
# Input: old_population - list of objects of MyCreature type that participated in the last simulation.  You
#                         can query the state of the creatures by using some built-in methods as well as any methods
#                         you decide to add to MyCreature class.  The length of the list is the size of
#                         the population.  You need to generate a new population of the same size.  Creatures from
#                         old population can be used in the new population - simulation will reset them to starting
#                         state.
#
# Returns: a list of MyCreature objects of the same length as the old_population.
def newPopulation(old_population):
    global numTurns
    global average_fitness

    nSurvivors = 0
    avgLifeTime = 0
    fitnessScore = 0
    mutationProb = 0.1
    elitismPercent = 0.05
    tournamentSelection = True

    # For each individual you can extract the following information left over
    # from evaluation to let you figure out how well individual did in the
    # simulation of the world: whether the creature is dead or not, how much
    # energy did the creature have a the end of simualation (0 if dead), tick number
    # of creature's death (if dead).  You should use this information to build
    # a fitness function, score for how the individual did
    for individual in old_population:

        # You can read the creature's energy at the end of the simulation.  It will be 0 if creature is dead
        energy = individual.getEnergy()

        # This method tells you if the creature died during the simulation
        dead = individual.isDead()

        # If the creature is dead, you can get its time of death (in turns)
        if dead:
            timeOfDeath = individual.timeOfDeath()
            avgLifeTime += timeOfDeath
        else:
            nSurvivors += 1
            avgLifeTime += numTurns
            timeOfDeath = numTurns
        individual.score = fitness_fn(timeOfDeath,energy,dead)
        fitnessScore += individual.score

    # Here are some statistics, which you may or may not find useful
    avgLifeTime = float(avgLifeTime)/float(len(population))
    avgScore = fitnessScore/len(population)
    average_fitness.append(avgScore)
    print("Simulation stats:")
    print("  Survivors          : %d out of %d" % (nSurvivors, len(population)))
    print("  Avg life time      : %.1f turns" % avgLifeTime)
    print("  Avg fitness score  : %.1f" % avgScore)

    # The information gathered above should allow you to build a fitness function that evaluates fitness of
    # every creature.  You should show the average fitness, but also use the fitness for selecting parents and
    # creating new creatures.

    #Elitism
    old_population.sort(key=lambda x: x.score, reverse=True)
    elite_percent = int(elitismPercent * len(old_population) + (int(elitismPercent * len(old_population))%2))
    elitism = old_population[:elite_percent]

    #Selection
    p = tournament_selection(old_population, elite_percent) if tournamentSelection else selection(old_population, fitnessScore, elite_percent)
    p = np.concatenate((elitism,p))

    #Crossover
    newChromosomes = []
    for x, y in zip(p[0::2], p[1::2]):
        children = recombine(x, y)
        newChromosomes.append(children[0])
        newChromosomes.append(children[1])

    #Mutation
    gene_pool = np.array(newChromosomes)
    gene_pool = gene_pool.reshape(-1, gene_pool.shape[-1])
    for x in range(len(newChromosomes)):
        newChromosomes[x] = mutate(newChromosomes[x], gene_pool, mutationProb)
        old_population[x].chromosome = newChromosomes[x]
    # Based on the fitness you should select individuals for reproduction and create a
    # new population.  At the moment this is not done, and the same population with the same number
    # of individuals
    new_population = [*old_population[len(elitism):],*elitism]

    return new_population


plt.close('all')
fh = plt.figure()

# Create the world.  Representaiton type choses the type of percept representation (there are three types to chose from);
# gridSize specifies the size of the world, repeatable parameter allows you to run the simulation in exactly same way.
w = World(worldType=worldType, gridSize=gridSize, repeatable=repeatableMode)

# Get the number of creatures in the world
numCreatures = w.maxNumCreatures()

# Get the number of creature percepts
numCreaturePercepts = w.numCreaturePercepts()

# Get the number of creature actions
numCreatureActions = w.numCreatureActions()

# Create a list of initial creatures - instantiations of the MyCreature class that you implemented
population = list()
for i in range(numCreatures):
    c = MyCreature(numCreaturePercepts, numCreatureActions)
    population.append(c)

# Pass the first population to the world simulator
w.setNextGeneration(population)

# Runs the simulation to evalute the first population
w.evaluate(numTurns)

# Show visualisation of initial creature behaviour
w.show_simulation(titleStr='Initial population', speed='normal')

for i in range(numGenerations):
    print("\nGeneration %d:" % (i+1))

    # Create a new population from the old one
    population = newPopulation(population)

    # Pass the new population to the world simulator
    w.setNextGeneration(population)

    # Run the simulation again to evalute the next population
    w.evaluate(numTurns)

    # Show visualisation of final generation
    if i == numGenerations-1:
        w.show_simulation(titleStr='Final population', speed='normal')

xi = np.arange(0,len(average_fitness))
slope, intercept, r_value, p_value, std_err = stats.linregress(xi,average_fitness)
line = slope*xi+intercept
plt.figure(1)
plt.plot(average_fitness)
plt.plot(line)
plt.title("Average Fitness for World {} | Grid:{}".format(worldType, gridSize))
plt.ylabel("Average Fitness")
plt.xlabel("Generations")
plt.show()