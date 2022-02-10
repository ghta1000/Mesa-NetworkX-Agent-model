import math
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from mesa.visualization.modules import TextElement
from .model import TumorModel, State, number_M, number_dead
import numpy as np

def network_portrayal(G):
    # The model ensures there is always 1 agent per node

    def node_color(agent):
        if agent.state is State.inflammatory:
            return '#0000FF'
        elif agent.state is State.CSC:
            return '#FF0000'
        elif agent.state is State.Dead:
            return '#808080'
        else:
            return '#9932CC'

    def edge_color(agent1, agent2):
        if agent1.state is State.Stem or agent2.state is State.Stem:
            return '#e8e8e8'
        return '#000000'

    def edge_width(agent1, agent2):
        if agent1.state is State.Stem or agent2.state is State.Stem:
            return 2
        return 1



    portrayal = dict()
    portrayal['nodes'] = [{'id': n_id,
                           'agent_id': n['agent'][0].unique_id,
                           'size': 2,
                           'color': node_color(n['agent'][0]),
                           }
                          for n_id, n in G.nodes(data=True)]

    portrayal['edges'] = [{'id': i,
                           'source': source,
                           'target': target,
                           'color': edge_color(G.node[source]['agent'][0], G.node[target]['agent'][0]),
                           'width': edge_width(G.node[source]['agent'][0], G.node[target]['agent'][0]),
                           }
                          for i, (source, target, _) in enumerate(G.edges(data=True))]

    return portrayal


network = NetworkModule(network_portrayal, 800, 800, library='d3')
chart = ChartModule([{"Label": 'CSC', 'Color': '#FF0000'},
                     {"Label": 'inflammatory Cell', 'Color': '#0000FF'},
                     {'Label': 'Stem', 'Color': '#9932CC'},
                     {"Label": 'Dead', 'Color': '#808080'}],
                     data_collector_name='datacollector')




class CSCRemainingElement(TextElement):
    def render(self, model):
        Cancer = number_M(model)
        return 'Number of inflamated cells: ' + str(Cancer)


class DeadCells(TextElement):
    def render(self, model):
        Dead = number_dead(model)
        return 'Number of Dead cells: ' + str(Dead)


class Density(TextElement):
        def render(self, model):
            Density = model.Density_Calculation()
            return 'Density: ' + str(Density)


text = CSCRemainingElement(), Density(), DeadCells()
n_slider = UserSettableParameter('slider', "Number", 100, 2, 200, 1)


model_params = {
    'num_nodes': UserSettableParameter('slider', 'Number of agents', 100, 100, 1000000, 1,
                                       description='Choose how many agents to include in the model'),
    'avg_node_degree': UserSettableParameter('slider', 'Avg Node Degree', 3, 3, 8, 1,
                                             description='Avg Node Degree'),
    'initial_Tumor_cell': UserSettableParameter('slider', 'Initial Tumor Cell', 1, 1, 1000, 1,
                                                   description='Initial Tumor Size'),
    'Angiogenesis_chance': UserSettableParameter('slider', 'Angiogenesis Chance', 0.1, 0.0, 1.0, 0.1,
                                                 description='Probability that vascular neighbor will be infected'),
    'Tumor_propagation_Frequency': UserSettableParameter('slider','Tumor propagation Frequency', 0.1, 0.0, 1.0, 0.1,
                                                   description='Frequency the nodes check whether they are infected '
                                                              ),
    'Angioprevention_chance': UserSettableParameter('slider', ' Angioprevention_chance', 0.1, 0.0, 1.0, 0.1,
                                                    description='Probability that switch Angiogenesis'),
}

server = ModularServer(TumorModel, [network, chart, *text], 'Tumor Model', model_params)


server.port = 8521