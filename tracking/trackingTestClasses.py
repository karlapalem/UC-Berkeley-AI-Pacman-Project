# trackingTestClasses.py
# ----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# trackingTestClasses.py
# ----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import testClasses
import busters
import layout
import bustersAgents
from game import Agent
from game import Actions
from game import Directions
import random
import time
import util
import json
import re
import copy
from util import manhattanDistance

class GameScoreTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(GameScoreTest, self).__init__(question, testDict)
        self.maxMoves = int(self.testDict['maxMoves'])
        self.inference = self.testDict['inference']
        self.layout_str = self.testDict['layout_str'].split('\n')
        self.numRuns = int(self.testDict['numRuns'])
        self.numWinsForCredit = int(self.testDict['numWinsForCredit'])
        self.numGhosts = int(self.testDict['numGhosts'])
        self.layout_name = self.testDict['layout_name']
        self.min_score = int(self.testDict['min_score'])
        self.observe_enable = self.testDict['observe'] == 'True'
        self.elapse_enable = self.testDict['elapse'] == 'True'

    def execute(self, grades, moduleDict, solutionDict):
        ghosts = [SeededRandomGhostAgent(i) for i in range(1,self.numGhosts+1)]
        pac = bustersAgents.GreedyBustersAgent(0, inference = self.inference, ghostAgents = ghosts, observeEnable = self.observe_enable, elapseTimeEnable = self.elapse_enable)
        #if self.inference == "ExactInference":
        #    pac.inferenceModules = [moduleDict['inference'].ExactInference(a) for a in ghosts]
        #else:
        #    print "Error inference type %s -- not implemented" % self.inference
        #    return

        stats = run(self.layout_str, pac, ghosts, self.question.getDisplay(), nGames=self.numRuns, maxMoves=self.maxMoves, quiet = False)
        aboveCount = [s >= self.min_score for s in stats['scores']].count(True)
        msg = "%s) Games won on %s with score above %d: %d/%d" % (self.layout_name, grades.currentQuestion, self.min_score, aboveCount, self.numRuns)
        grades.addMessage(msg)
        if aboveCount >= self.numWinsForCredit:
            grades.assignFullCredit()
            return self.testPass(grades)
        else:
            return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# You must win at least %d/10 games with at least %d points' % (self.numWinsForCredit, self.min_score))
        handle.close()

    def createPublicVersion(self):
        pass

class ZeroWeightTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(ZeroWeightTest, self).__init__(question, testDict)
        self.maxMoves = int(self.testDict['maxMoves'])
        self.inference = self.testDict['inference']
        self.layout_str = self.testDict['layout'].split('\n')
        self.numGhosts = int(self.testDict['numGhosts'])
        self.observe_enable = self.testDict['observe'] == 'True'
        self.elapse_enable = self.testDict['elapse'] == 'True'
        self.ghost = self.testDict['ghost']
        self.seed = int(self.testDict['seed'])

    def execute(self, grades, moduleDict, solutionDict):
        random.seed(self.seed)
        inferenceFunction = getattr(moduleDict['inference'], self.inference)
        ghosts = [globals()[self.ghost](i) for i in range(1, self.numGhosts+1)]
        if self.inference == 'MarginalInference':
            moduleDict['inference'].jointInference = moduleDict['inference'].JointParticleFilter()
        disp = self.question.getDisplay()
        pac = ZeroWeightAgent(inferenceFunction, ghosts, grades, self.seed, disp, elapse=self.elapse_enable, observe=self.observe_enable)
        if self.inference == "ParticleFilter":
            for pfilter in pac.inferenceModules: pfilter.setNumParticles(5000)
        elif self.inference == "MarginalInference":
            moduleDict['inference'].jointInference.setNumParticles(5000)
        run(self.layout_str, pac, ghosts, disp, maxMoves = self.maxMoves)
        if pac.getReset():
            grades.addMessage('%s) successfully handled all weights = 0' % grades.currentQuestion)
            return self.testPass(grades)
        else:
            grades.addMessage('%s) error handling all weights = 0' % grades.currentQuestion)
            return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This test checks that you successfully handle the case when all particle weights are set to 0\n')
        handle.close()

    def createPublicVersion(self):
        self.testDict['seed'] = '188'
        self.seed = 188

class DoubleInferenceAgentTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(DoubleInferenceAgentTest, self).__init__(question, testDict)
        self.seed = int(self.testDict['seed'])
        self.layout_str = self.testDict['layout'].split('\n')
        self.observe = (self.testDict['observe'] == "True")
        self.elapse = (self.testDict['elapse'] == "True")
        self.checkUniform = (self.testDict['checkUniform'] == 'True')
        self.maxMoves = int(self.testDict['maxMoves'])
        self.numGhosts = int(self.testDict['numGhosts'])
        self.inference = self.testDict['inference']
        self.errorMsg = self.testDict['errorMsg']
        self.L2Tolerance = float(self.testDict['L2Tolerance'])
        self.ghost = self.testDict['ghost']

    def execute(self, grades, moduleDict, solutionDict):
        random.seed(self.seed)
        lines = solutionDict['correctActions'].split('\n')
        moves = []
        # Collect solutions
        for l in lines:
            m = re.match('(\d+) (\w+) (.*)', l)
            moves.append((m.group(1), m.group(2), eval(m.group(3))))

        inferenceFunction = getattr(moduleDict['inference'], self.inference)

        ghosts = [globals()[self.ghost](i) for i in range(1, self.numGhosts+1)]
        if self.inference == 'MarginalInference':
            moduleDict['inference'].jointInference = moduleDict['inference'].JointParticleFilter()

        disp = self.question.getDisplay()
        pac = DoubleInferenceAgent(inferenceFunction, moves, ghosts, grades, self.seed, disp, elapse=self.elapse, observe=self.observe, L2Tolerance=self.L2Tolerance, checkUniform = self.checkUniform)
        if self.inference == "ParticleFilter":
            for pfilter in pac.inferenceModules: pfilter.setNumParticles(5000)
        elif self.inference == "MarginalInference":
            moduleDict['inference'].jointInference.setNumParticles(5000)
        run(self.layout_str, pac, ghosts, disp, maxMoves=self.maxMoves)
        msg = self.errorMsg % pac.errors
        grades.addMessage(("%s) " % (grades.currentQuestion))+msg)
        if pac.errors == 0:
            grades.addPoints(2)
            return self.testPass(grades)
        else:
            return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        random.seed(self.seed)
        if self.inference == 'ParticleFilter':
            self.inference = 'ExactInference'  # use exact inference to generate solution
        inferenceFunction = getattr(moduleDict['inference'], self.inference)

        ghosts = [globals()[self.ghost](i) for i in range(1, self.numGhosts+1)]
        if self.inference == 'MarginalInference':
            moduleDict['inference'].jointInference = moduleDict['inference'].JointParticleFilter()
            moduleDict['inference'].jointInference.setNumParticles(5000)

        pac = InferenceAgent(inferenceFunction, ghosts, self.seed, elapse=self.elapse, observe=self.observe)
        run(self.layout_str, pac, ghosts, self.question.getDisplay(), maxMoves=self.maxMoves)
        # run our gold code here and then write it to a solution file
        answerList = pac.answerList
        handle = open(filePath, 'w')
        handle.write('# move_number action likelihood_dictionary\n')
        handle.write('correctActions: """\n')
        for (moveNum, move, dists) in answerList:
            handle.write('%s %s [' % (moveNum, move))
            for dist in dists:
                handle.write('{')
                for key in dist:
                    handle.write('%s: %s, ' % (key, dist[key]))
                handle.write('}, ')
            handle.write(']\n')
        handle.write('"""\n')
        handle.close()

    def createPublicVersion(self):
        self.testDict['seed'] = '188'
        self.seed = 188

def run(layout_str, pac, ghosts, disp, nGames = 1, name = 'games', maxMoves=-1, quiet = True):
    "Runs a few games and outputs their statistics."

    starttime = time.time()
    lay = layout.Layout(layout_str)

    #print '*** Running %s on' % name, layname,'%d time(s).' % nGames
    games = busters.runGames(lay, pac, ghosts, disp, nGames, maxMoves)

    #print '*** Finished running %s on' % name, layname, 'after %d seconds.' % (time.time() - starttime)

    stats = {'time': time.time() - starttime, \
      'wins': [g.state.isWin() for g in games].count(True), \
      'games': games, 'scores': [g.state.getScore() for g in games]}
    statTuple = (stats['wins'], len(games), sum(stats['scores']) * 1.0 / len(games))
    if not quiet:
        print '*** Won %d out of %d games. Average score: %f ***' % statTuple
    return stats

class InferenceAgent(bustersAgents.BustersAgent):
    "Tracks ghosts and compares to reference inference modules, while moving randomly"

    def __init__( self, inference, ghostAgents, seed, elapse=True, observe=True, burnIn=0):
        self.inferenceModules = [inference(a) for a in ghostAgents]
        self.elapse = elapse
        self.observe = observe
        self.burnIn = burnIn
        self.numMoves = 0
        #self.rand = rand
        # list of tuples (move_num, move, [dist_1, dist_2, ...])
        self.answerList = []
        self.seed = seed

    def final(self, gameState):
        distributionList = []
        self.numMoves += 1
        for index,inf in enumerate(self.inferenceModules):
            if self.observe:
                inf.observeState(gameState)
            self.ghostBeliefs[index] = inf.getBeliefDistribution()
            beliefCopy = copy.deepcopy(self.ghostBeliefs[index])
            distributionList.append(beliefCopy)
        self.answerList.append((self.numMoves, None, distributionList))
        random.seed(self.seed + self.numMoves)

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        for inference in self.inferenceModules: inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True
        self.answerList.append((self.numMoves,None,copy.deepcopy(self.ghostBeliefs)))

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        distributionList = []
        self.numMoves += 1
        for index,inf in enumerate(self.inferenceModules):
            if self.elapse:
                if not self.firstMove: inf.elapseTime(gameState)
            self.firstMove = False
            if self.observe:
                inf.observeState(gameState)
            self.ghostBeliefs[index] = inf.getBeliefDistribution()
            beliefCopy = copy.deepcopy(self.ghostBeliefs[index])
            distributionList.append(beliefCopy)
        action = random.choice([a for a in gameState.getLegalPacmanActions() if a != 'STOP'])
        self.answerList.append((self.numMoves, action, distributionList))
        random.seed(self.seed + self.numMoves)
        return action


class ZeroWeightAgent(bustersAgents.BustersAgent):
    "Tracks ghosts and compares to reference inference modules, while moving randomly"

    def __init__( self, inference, ghostAgents, grades, seed, disp, elapse=True, observe=True ):
        self.inferenceModules = [inference(a) for a in ghostAgents]
        self.elapse = elapse
        self.observe = observe
        self.grades = grades
        self.numMoves = 0
        self.seed = seed
        self.display = disp
        self.reset = False

    def final(self, gameState):
        pass

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        for inference in self.inferenceModules: inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        newBeliefs = [None] * len(self.inferenceModules)
        self.numMoves += 1
        for index,inf in enumerate(self.inferenceModules):
            if self.elapse:
                if not self.firstMove: inf.elapseTime(gameState)
            self.firstMove = False
            if self.observe:
                inf.observeState(gameState)
            newBeliefs[index] = inf.getBeliefDistribution()
        self.checkReset(newBeliefs, self.ghostBeliefs)
        self.ghostBeliefs = newBeliefs
        self.display.updateDistributions(self.ghostBeliefs)
        random.seed(self.seed + self.numMoves)
        action = random.choice([a for a in gameState.getLegalPacmanActions() if a != 'STOP'])
        return action

    def checkReset(self, newBeliefs, oldBeliefs):
        for i in range(len(newBeliefs)):
            newKeys = filter(lambda x: newBeliefs[i][x] != 0, newBeliefs[i].keys())
            oldKeys = filter(lambda x: oldBeliefs[i][x] != 0, oldBeliefs[i].keys())
            if len(newKeys) > len(oldKeys):
                self.reset = True

    def getReset(self):
        return self.reset


class DoubleInferenceAgent(bustersAgents.BustersAgent):
    "Tracks ghosts and compares to reference inference modules, while moving randomly"

    def __init__( self, inference, refSolution, ghostAgents, grades, seed, disp, elapse=True, observe=True, L2Tolerance=0.2, burnIn=0, checkUniform = False):
        self.inferenceModules = [inference(a) for a in ghostAgents]
        self.refSolution = refSolution
        self.elapse = elapse
        self.observe = observe
        self.grades = grades
        self.L2Tolerance = L2Tolerance
        self.errors = 0
        self.burnIn = burnIn
        self.numMoves = 0
        self.seed = seed
        self.display = disp
        self.checkUniform = checkUniform

    def final(self, gameState):
        self.numMoves += 1
        moveNum,action,dists = self.refSolution[self.numMoves]
        for index,inf in enumerate(self.inferenceModules):
            if self.observe:
                inf.observeState(gameState)
            self.ghostBeliefs[index] = inf.getBeliefDistribution()
            if self.numMoves >= self.burnIn:
                self.distCompare(self.ghostBeliefs[index], dists[index])
        self.display.updateDistributions(self.ghostBeliefs)
        random.seed(self.seed + self.numMoves)
        if not self.display.checkNullDisplay():
            time.sleep(3)

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        for inference in self.inferenceModules: inference.initialize(gameState)
        moveNum,action,dists = self.refSolution[self.numMoves]
        for index,inf in enumerate(self.inferenceModules):
            self.distCompare(inf.getBeliefDistribution(), dists[index])
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        self.numMoves += 1
        moveNum,action,dists = self.refSolution[self.numMoves]
        for index,inf in enumerate(self.inferenceModules):
            if self.elapse:
                if not self.firstMove: inf.elapseTime(gameState)
            self.firstMove = False
            if self.observe:
                inf.observeState(gameState)
            self.ghostBeliefs[index] = inf.getBeliefDistribution()
            if self.numMoves >= self.burnIn: self.distCompare(self.ghostBeliefs[index], dists[index])
        self.display.updateDistributions(self.ghostBeliefs)
        random.seed(self.seed + self.numMoves)
        return action

    def distCompare(self, dist, refDist):
        "Compares two distributions"
        # copy and prepare distributions
        dist = dist.copy()
        refDist = refDist.copy()
        for key in set(refDist.keys() + dist.keys()):
            if not key in dist.keys():
                dist[key] = 0.0
            if not key in refDist.keys():
                refDist[key] = 0.0
        # calculate l2 difference
        l2 = 0
        for k in refDist.keys():
            l2 += (dist[k] - refDist[k]) ** 2
        if l2 > self.L2Tolerance:
            if self.errors == 0:
                t = (self.grades.currentQuestion, self.numMoves, l2)
                summary = "%s) Distribution deviated at move %d by %0.4f (squared norm) from the correct answer.\n" % t
                header = '%10s%5s%-25s%-25s\n' % ('key:', '', 'student', 'reference')
                detail = '\n'.join(map(lambda x: '%9s:%5s%-25s%-25s' % (x, '', dist[x], refDist[x]), set(dist.keys() + refDist.keys())))
                self.grades.fail('%s%s%s' % (summary, header, detail))
            self.errors += 1
        # check for uniform distribution if necessary
        if self.checkUniform:
            if abs(max(dist.values()) - max(refDist.values())) > .0025:
                if self.errors == 0:
                    self.grades.fail('%s) Distributions do not have the same max value and are therefore not uniform.\n\tstudent max: %f\n\treference max: %f' % (self.grades.currentQuestion, max(dist.values()), max(refDist.values())))
                    self.errors += 1

class SeededRandomGhostAgent(Agent):
    def __init__(self, index):
        self.index = index;

    def getAction(self, state):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        if len(dist) == 0:
            return Directions.STOP
        else:
            action = self.sample( dist )
            return action

    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist

    def sample(self, distribution, values = None):
        if type(distribution) == util.Counter:
            items = distribution.items()
            distribution = [i[1] for i in items]
            values = [i[0] for i in items]
        if sum(distribution) != 1:
            distribution = normalize(distribution)
        choice = random.random()
        i, total= 0, distribution[0]
        while choice > total:
            i += 1
            total += distribution[i]
        return values[i]

class GoSouthAgent(Agent):
    def __init__(self, index):
        self.index = index;

    def getAction(self, state):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ):
            dist[a] = 1.0
        if Directions.SOUTH in dist.keys():
            dist[Directions.SOUTH] *= 2
        dist.normalize()
        if len(dist) == 0:
            return Directions.STOP
        else:
            action = self.sample( dist )
            return action

    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ):
            dist[a] = 1.0
        if Directions.SOUTH in dist.keys():
            dist[Directions.SOUTH] *= 2
        dist.normalize()
        return dist

    def sample(self, distribution, values = None):
        if type(distribution) == util.Counter:
            items = distribution.items()
            distribution = [i[1] for i in items]
            values = [i[0] for i in items]
        if sum(distribution) != 1:
            distribution = util.normalize(distribution)
        choice = random.random()
        i, total= 0, distribution[0]
        while choice > total:
            i += 1
            total += distribution[i]
        return values[i]

class DispersingSeededGhost( Agent):
    "Chooses an action that distances the ghost from the other ghosts with probability spreadProb."
    def __init__( self, index, spreadProb=0.5):
        self.index = index
        self.spreadProb = spreadProb

    def getAction(self, state):
        dist = self.getDistribution(state);
        if len(dist) == 0:
            return Directions.STOP
        else:
            action = self.sample( dist )
            return action

    def getDistribution( self, state ):
        ghostState = state.getGhostState( self.index )
        legalActions = state.getLegalActions( self.index )
        pos = state.getGhostPosition( self.index )
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5
        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]

        # get other ghost positions
        others = [i for i in range(1,state.getNumAgents()) if i != self.index]
        for a in others: assert state.getGhostState(a) != None, "Ghost position unspecified in state!"
        otherGhostPositions = [state.getGhostPosition(a) for a in others if state.getGhostPosition(a)[1] > 1]

        # for each action, get the sum of inverse squared distances to the other ghosts
        sumOfDistances = []
        for pos in newPositions:
            sumOfDistances.append( sum([(1+manhattanDistance(pos, g))**(-2) for g in otherGhostPositions]) )

        bestDistance = min(sumOfDistances)
        numBest = [bestDistance == dist for dist in sumOfDistances].count(True)
        distribution = util.Counter()
        for action, distance in zip(legalActions, sumOfDistances):
            if distance == bestDistance: distribution[action] += self.spreadProb / numBest
            distribution[action] += (1 - self.spreadProb) / len(legalActions)
        return distribution

    def sample(self, distribution, values = None):
        if type(distribution) == util.Counter:
            items = distribution.items()
            distribution = [i[1] for i in items]
            values = [i[0] for i in items]
        if sum(distribution) != 1:
            distribution = util.normalize(distribution)
        choice = random.random()
        i, total= 0, distribution[0]
        while choice > total:
            i += 1
            total += distribution[i]
        return values[i]
