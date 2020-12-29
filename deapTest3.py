import preload
import numpy
import csv
import math
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
import deap
import random
import matplotlib.pyplot as plt

import pandas as pd
# pushing
from pandas import DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import exp

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
random.seed()
toolbox.register("attr_bool", random.randint, 0, 50)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_bool, 4)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def movingAveRule(price,dateC,dateP,f,s,panda):

    dates = panda["Date"]
    fast = panda["SMA "+str(f)]
    slow = panda["SMA "+str(s)]

    fastDict = dict(zip(dates, fast))
    slowDict = dict(zip(dates, slow))

    if fastDict[dateP] < slowDict[dateP] and fastDict[dateC] > slowDict[dateC]:
        return 1
    elif fastDict[dateP] > slowDict[dateP] and fastDict[dateC] < slowDict[dateC]:
        return -1
    else:
        return 0

# the goal ('fitness') function to be maximized
def emaRule(price,dateC,dateP,f,s,panda):

    dates = panda["Date"]
    fast = panda["EMA "+str(f)]
    slow = panda["EMA "+str(s)]

    fastDict = dict(zip(dates, fast))
    slowDict = dict(zip(dates, slow))

    if fastDict[dateP] < slowDict[dateP] and fastDict[dateC] > slowDict[dateC]:
        return 1
    elif fastDict[dateP] > slowDict[dateP] and fastDict[dateC] < slowDict[dateC]:
        return -1
    else:
        return 0

def simulation(x):
    # 0-23 in these lists. So 24
    bounds = []
    for i in range(1,52):
        bounds.append(float(i))
    #fast = [2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0]
    #slow = [20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0]
    f = bounds[int(x[0])]
    s = bounds[int(x[1])]
    f1 = bounds[int(x[2])]
    s1 = bounds[int(x[3])]
    #print(x)

    startDate = '2019-01-01'
    endDate = '2020-01-01'
    amount = 1000
    shares = 0
    balance = 1000
    returns = 0
    equity = 0
    startDay = datetime.strptime(startDate,'%Y-%m-%d')
    endDay = datetime.strptime(endDate,'%Y-%m-%d')

    panda = pd.read_csv('MSFT.csv')
    dates = panda["Date"]
    prices = panda["Close"]
    combined = dict(zip(dates, round(prices,2)))
    iter = 0
    startCount = 0
    bhBalance = 0
    bhShares = 0

    for date, price in combined.items(): # Need to make within the sim time frame
        # to start the sim from the start date
        if datetime.strptime(date,'%Y-%m-%d') < startDay and startCount == 0:
            continue
        # calculating the b&h strategy at start date
        elif startCount == 0:
            startD = date
            startCount = 1
            bhShares = amount / combined[date]

        if iter == 0:
            oldDate = date
            iter += 1
            continue

        if movingAveRule(price,date,oldDate,f,s,panda) == 1 and emaRule(price,date,oldDate,f1,s1,panda) == 1:
            shares = amount/price
            balance -= amount

        elif movingAveRule(price,date,oldDate,f,s,panda) == -1 and emaRule(price,date,oldDate,f1,s1,panda) == -1 and shares > 0:
            balance = shares*price
            profit = balance - amount
            returns += profit

        elif shares != 0:
            equity = price*shares

        # to end the sim at the end date
        if datetime.strptime(date,'%Y-%m-%d') >= endDay:
            bhBalance = bhShares*combined[oldDate]
            break

        oldDate = date
        answer = (((returns+equity)-amount)/amount)*100

    #print("GETS: ",round(answer,2))

    return round(answer,2), returns+equity

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", simulation)

# register the crossover operator
toolbox.register("mate", tools.cxOnePoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutUniformInt, low=0, up=50, indpb=0.2)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament,tournsize=3)

#----------


def draw_colormap (cmap_name,width=60,height=4,fontsize=50):
    """
    cmap_name is a string.
    """
    fig = plt.figure(figsize=(width,height))
    ax = plt.gca()
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['font.size'] = fontsize
    # Divide the interval from 0 to 1 into 256 parts.
    gradient = numpy.linspace(0, 70, 256)
    # The imshow function actually displays images,
    # but you can pass it an array.
    # In either case, the function wants 2D info.
    # We give it an array with 2 rows,
    # make the top half and bottom half  of the image the same.
    # We'll vertically stretch this very skinny image with aspect 'auto'.
    # And the value of gradient will change the color value from left to right.
    gradient = numpy.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto',cmap=plt.get_cmap(cmap_name))
    # Show the number values associated with color on x-axis.
    # The xvalues that imshow uses are pixel numbers, integers from 0 to 256.
    ax.set_xticks(numpy.linspace(0,256,11))
    # We'll label those with the inputs to the color map, numbers from 0 to 1
    ax.set_xticklabels(numpy.linspace(0,70,11))
    ax.set_yticklabels([])
    plt.show()


def main(ngen,popu,cxpb,mutpb,graph=True,parallelCoord=False):
    random.seed()
    NGEN = ngen
    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=popu)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    fbest = numpy.ndarray((NGEN+1,1))

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = cxpb, mutpb

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    gen = 0
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(evals=len(pop), gen=gen, **record)
    fbest[gen] = hof[0].fitness.values
    print("  Max %s" % max(fits))
    # fast = []
    # slow = []
    # for i in pop:
    #     fast.append(i[0])
    #     slow.append(i[1])
    # plt.plot(fast, slow,'x', label="Average fitness")
    # plt.show()
    dictOfAll = {}
    dictOfMax = {}
    count = 0
    # Begin the evolution
    while max(fits) < 8000 and gen < NGEN:
        # A new generation
        gen = gen + 1
        print("-- Generation %i --" % gen)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        for child in offspring:
            dictOfAll[count] = child
            count += 1

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)

        best_ind = tools.selBest(pop, 1)[0]

        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Best params %s" % best_ind)

        dictOfMax[gen] = best_ind

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(evals=len(pop), gen=gen, **record)
        fbest[gen] = hof[0].fitness.values

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    if graph == True:
        x = list(range(0, NGEN+1))
        avg, max_, min_ = logbook.select("avg", "max", "min")

        avg1 = numpy.array(avg)

        plt.plot(x, avg,'x', label="Average fitness")
        plt.plot(x, fbest,'ro',label="Max fitness")
        plt.legend()
        plt.xlabel("Number of generations") # Add ", fontsize = #" to control fontsize
        plt.ylabel("% increase in profit / fitness")
        title = "Performance of GA (Population of "+str(popu)+" individuals, Crossover rate = "+str(cxpb)+", Mutation rate = "+str(mutpb)+")"
        plt.title(title)
        plt.show()

    if parallelCoord == True:
        print("Making Parallel Coordinates")
        cmap_name = "YlOrRd"
        draw_colormap(cmap_name)
        d = []
        for k,v in dictOfAll.items():
            d.append(
                {
                    'Fast SMA': v[0],
                    'Slow SMA': v[1],
                    'Fast EMA': v[2],
                    'Slow EMA': v[3]
                }
            )

        pandaDF = pd.DataFrame(d)
        print (pandaDF)

        plt.figure(figsize=(20,10))
        mpl.rcParams['xtick.labelsize'] = 14
        mpl.rcParams['font.size'] = 14
        for i in range(0,count):
            dataRow = pandaDF.iloc[i,0:4]
            labelColor = i/count
            dataRow.plot(color=plt.cm.YlOrRd(labelColor), alpha=0.5)
            #Use the dsame color for every exemplar. Dull.
            #dataRow.plot(color=plot.cm.RdYlBu(.1), alpha=0.5)

        plt.xlabel("Parameter Index")
        plt.ylabel(("Parameter Values"))
        plt.show()


if __name__ == "__main__":
    main(70,100,0.5,0.3,parallelCoord=True)
