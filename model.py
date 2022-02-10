import random
import math
from enum import Enum
import networkx as nx
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
import time
import numpy as np
from networkx.readwrite import json_graph
import json
import csv as csv




class State(Enum):
    CSC = 1# Tumor Cells
    inflammatory = 2
    Stem = 3.# Normal Cells
    Dead = 0# Apoptotic Cells



def number_state(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if a.state is state])

def number_CSC(model):
    return number_state(model, State.CSC)

def number_Stem(model):
    return number_state(model, State.Stem)

def number_M(model):
    return number_state(model, State.inflammatory)

def number_dead(model):
    return number_state(model, State.Dead)

class TumorModel(Model):
    """A Tumor model with some number of agents"""

    def __init__(self, num_nodes, avg_node_degree, initial_Tumor_cell, Angiogenesis_chance, Tumor_propagation_Frequency,
                  Angioprevention_chance):
        self.GR = nx.read_graphml("GrowPatterm.1.1.graphml")
        self.num_nodes = nx.number_of_nodes(self.GR)
        #self.redirected_nodes = int(((2 * 12000) *2)/2)
        self.prob = avg_node_degree / self.num_nodes
        #self.G = nx.gnr_graph(self.redirected_nodes, self.prob, create_using=self.GR, seed=None)
        self.G = nx.powerlaw_cluster_graph(n=self.num_nodes, m=2, p=self.prob, seed=None)
        #self.G = nx.random_powerlaw_tree(n=self.num_nodes, gamma=3, seed=None, tries=10)
        self.GP = nx.write_graphml(self.G, "Growcluster.1.1.graphml")
        self.D = nx.density(self.G)
        self.O = nx.write_graphml(self.G,"OutputC1.graphml", encoding='utf-8', prettyprint=True)
        self.M = nx.to_numpy_matrix(self.G)
        self.C = dict(nx.betweenness_centrality(self.G, k=None, normalized=True, weight=None, endpoints=False, seed=None))
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.initial_Tumor_cell = initial_Tumor_cell if initial_Tumor_cell <= num_nodes else num_nodes
        self.Angiogenesis_chance = Angiogenesis_chance
        self.Tumor_propagation_Frequency = Tumor_propagation_Frequency
        self.Angioprevention_chance = Angioprevention_chance
        self.datacollector = DataCollector({"CSC": number_CSC,
                                            "inflammatory": number_M,
                                            "Dead": number_dead,
                                            "Stem": number_Stem})





        t_start = time.time()

        # Create agents
        for i, node in enumerate(self.G.nodes()):

            a = TumorAgent(i, self, self.Angiogenesis_chance, self.Tumor_propagation_Frequency, self.initial_Tumor_cell,
                           self.Angioprevention_chance)

            self.schedule.add(a)
            # Add the agent to the node
            self.grid.place_agent(a, node)



         # Inflammate some nodes
        Inflammated_nodes = random.sample(self.G.nodes(), self.initial_Tumor_cell)
        for a in self.grid.get_cell_list_contents(Inflammated_nodes):
            a.state = State.CSC


        self.running = True
        self.datacollector.collect(self)


        np.savetxt('Matrix.out', self.M, fmt='%s')
        w = csv.writer(open("Dictionary6.2.csv", "w"))

        for key, val in dict.items(self.C):
            w.writerow([key, val])


        t_end = time.time()
        print("[ Lapse: {} seconds".format(t_end - t_start))

    def Stem_inflammatory_ratio(self):
        try:
          return self.C
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
    def __init__(self, unique_id, model, initial_state, Angiogenesis_chance, Tumor_propagation_Frequency,
                  Angioprevention_chance):
        super().__init__(unique_id, model)

        self.state = initial_state

        self.Angiogenesis_chance = Angiogenesis_chance
        self.Tumor_propagation_Frequency = Tumor_propagation_Frequency
        self.Angioprevention_chance = Angioprevention_chance

    def try_to_inflammate_neighbors(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        inflammatory_neighbors = [agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes) if
                                 agent.state is State.CSC]

        for a in inflammatory_neighbors:
            if random.random() < self.Angiogenesis_chance:
                a.state = State.inflammatory

    def try_gain_Quiescent(self):
             if random.random() < self.Angioprevention_chance:
                self.state = State.Dead

    def try_check_situation(self):
        if random.random() > self.Tumor_propagation_Frequency:
            self.try_gain_Quiescent
        else:
            self.try_to_inflammate_neighbors()

    def step(self):
            if self.state is State.Stem:
                self.try_to_inflammate_neighbors()
            self.try_check_situation()



