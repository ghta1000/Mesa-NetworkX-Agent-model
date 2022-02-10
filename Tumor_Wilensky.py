import random
import math
from enum import Enum
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
import time
import networkx as nx
import numpy as np


class State(Enum):
    METASTAISE = 1  # Tumor Cells
    Transitory = 2  # daughter cell
    Stem = 3  # Normal Cells
    Dead = 0# Apoptotic Cells



def number_state(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if a.state is state])

def number_METASTAISE(model):
    return number_state(model, State.METASTAISE)


def number_Stem(model):
    return number_state(model, State.Stem)


def number_M(model):
    return number_state(model, State.Transitory)

def number_dead(model):
    return number_state(model, State.Dead)

class TumorModel(Model):
    """A Tumor model with some number of agents"""

    def __init__(self, num_nodes, avg_node_degree, initial_outbreak_size, Angiogenesis_chance, Mitosis_frequency,
                 recovery_chance, Angioprevention_chance):

        self.num_nodes = num_nodes
        prob = avg_node_degree / self.num_nodes
        # self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=prob)
        self.G = nx.fast_gnp_random_graph(n=self.num_nodes, p=prob, directed=True)
        self.M = nx.to_numpy_matrix(self.G)
        self.M = nx.write_adjlist(self.G, "test8.adjlist")
        self.N = nx.write_graphml(self.G, "testCluster1.graphml")
        self.F = nx.write_gml(self.G, "path.to.file.8")

        self.C = nx.betweenness_centrality(self.G, k=None, normalized=True, weight=None, endpoints=False, seed=None)
        self.D = nx.density(self.G)
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.initial_outbreak_size = initial_outbreak_size if initial_outbreak_size <= num_nodes else num_nodes
        self.Angiogenesis_chance = Angiogenesis_chance
        self.Mitosis_frequency = Mitosis_frequency
        self.recovery_chance = recovery_chance
        self.Angioprevention_chance = Angioprevention_chance

        self.datacollector = DataCollector({"METASTAISE": number_METASTAISE,
                                            "Transitory": number_M,
                                            "Dead": number_dead,
                                            "Stem": number_Stem})


        t_start = time.time()
        # Create agents
        for i, node in enumerate(self.G.nodes()):

            a = TumorAgent(i, self, State.METASTAISE, self.Angiogenesis_chance, self.Mitosis_frequency,
                           self.recovery_chance, self.Angioprevention_chance)
            self.schedule.add(a)
            # Add the agent to the node
            self.grid.place_agent(a, node)

        t_end = time.time()

        print("[ Lapse: {} seconds".format(t_end - t_start))

        # Transient some nodes
        infected_nodes = random.sample(self.G.nodes(), self.initial_outbreak_size)
        for a in self.grid.get_cell_list_contents(infected_nodes):
            a.state = State.Transitory

        self.running = True
        self.datacollector.collect(self)
        #np.savetxt('TV.txt', self.C, fmt='%10.5f', delimiter=',', newline='\n', header='', footer='', comments='# ', encoding=None)


    def Stem_Transitory_ratio(self):
        try:
            return number_state(self, State.Stem) / number_state(self, State.METASTAISE)
        except ZeroDivisionError:
            return math.inf

    def Density_Calculation(self):
            try:
                return self.D
            except ZeroDivisionError:
                return math.inf

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)


    def run_model(self, n):
        print("model run")
        for i in range(n):
            self.step()



class TumorAgent(Agent):
    def __init__(self, unique_id, model, initial_state, Angiogenesis_chance, Mitosis_frequency,
                 recovery_chance, Angioprevention_chance):
        super().__init__(unique_id, model)

        self.state = initial_state

        self.Angiogenesis_chance = Angiogenesis_chance
        self.Mitosis_frequency = Mitosis_frequency
        self.recovery_chance = recovery_chance
        self.Angioprevention_chance = Angioprevention_chance

    def try_to_infect_neighbors(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        transitory_neighbors = [agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes) if
                                 agent.state is State.METASTAISE]
        for a in transitory_neighbors:
            if random.random() < self.Angiogenesis_chance:
                a.state = State.Stem
            else:
                a.state = State.Transitory

    def try_gain_Quiescent(self):
        if random.random() < self.Angioprevention_chance:
            self.state = State.Dead
        else:
            self.state = State.Transitory
            self.try_to_infect_neighbors()


    def try_kill_cancer(self):
        # Try to kill
        if random.random() > self.recovery_chance:
            # Failed
            self.state = State.METASTAISE
            self.try_gain_Quiescent()
        else:
            # Success
            self.state = State.Stem

    def try_check_situation(self):
        if random.random() > self.Mitosis_frequency:
            # Checking...
            if self.state is State.Transitory:
                self.try_kill_cancer()

    def step(self):
        if self.state is State.Transitory:
            self.try_to_infect_neighbors()
        self.try_check_situation()
