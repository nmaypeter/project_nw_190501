from Diffusion import *
from Evaluation import *


def selectDegreeSeed(d_dict):
    # -- get the node with highest degree --
    mep = (-1, '-1')
    max_degree = -1
    while mep[1] == '-1':
        while max_degree == -1:
            for deg in list(d_dict.keys()):
                if int(deg) > max_degree:
                    max_degree = int(deg)

            if max_degree == -1:
                return mep

            if d_dict[str(max_degree)] == set():
                del d_dict[str(max_degree)]
                max_degree = -1

        if d_dict[str(max_degree)] == set():
            del d_dict[str(max_degree)]
            max_degree = -1
            continue

        mep = choice(list(d_dict[str(max_degree)]))
        d_dict[str(max_degree)].remove(mep)

    return mep


class SeedSelectionHD:
    def __init__(self, g_dict, s_c_dict, prod_list):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)

    def constructDegreeDict(self, data_name):
        # -- display the degree and the nodes with the degree --
        ### d_dict: (dict) the degree and the nodes with the degree
        ### d_dict[deg]: (set) the set for deg-degree nodes
        d_dict = {}
        with open(IniGraph(data_name).data_degree_path) as f:
            for line in f:
                (i, deg) = line.split()
                if deg == '0':
                    continue
                for k in range(self.num_product):
                    if deg in d_dict:
                        d_dict[deg].add((k, i))
                    else:
                        d_dict[deg] = {(k, i)}
        f.close()

        return d_dict

    def constructExpendDegreeDict(self):
        # -- display the degree and the nodes with the degree --
        ### d_dict: (dict) the degree and the nodes with the degree
        ### d_dict[deg]: (set) the set for deg-degree nodes
        d_dict = {}
        for i in self.graph_dict:
            i_set = {i}
            for ii in self.graph_dict[i]:
                if ii not in i_set:
                    i_set.add(ii)
            for ii in self.graph_dict[i]:
                if ii in self.graph_dict:
                    for iii in self.graph_dict[ii]:
                        if iii not in i_set:
                            i_set.add(iii)

            deg = str(len(i_set))
            for k in range(self.num_product):
                if deg in d_dict:
                    d_dict[deg].add((k, i))
                else:
                    d_dict[deg] = {(k, i)}

        return d_dict


class SeedSelectionHDPW:
    def __init__(self, g_dict, s_c_dict, prod_list, pw_list):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### pw_list: (list) the product weight list
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.pw_list = pw_list

    def constructDegreeDict(self, data_name):
        # -- display the degree and the nodes with the degree --
        ### d_dict: (dict) the degree and the nodes with the degree
        ### d_dict[deg]: (set) the set for deg-degree nodes
        d_dict = {}
        with open(IniGraph(data_name).data_degree_path) as f:
            for line in f:
                (i, deg) = line.split()
                if deg == '0':
                    continue
                for k in range(self.num_product):
                    deg = str(round(float(deg) * self.pw_list[k]))
                    if deg in d_dict:
                        d_dict[deg].add((k, i))
                    else:
                        d_dict[deg] = {(k, i)}
        f.close()

        return d_dict

    def constructExpendDegreeDict(self):
        # -- display the degree and the nodes with the degree --
        ### d_dict: (dict) the degree and the nodes with the degree
        ### d_dict[deg]: (set) the set for deg-degree nodes
        d_dict = {}
        for i in self.graph_dict:
            i_set = {i}
            for ii in self.graph_dict[i]:
                if ii not in i_set:
                    i_set.add(ii)
            for ii in self.graph_dict[i]:
                if ii in self.graph_dict:
                    for iii in self.graph_dict[ii]:
                        if iii not in i_set:
                            i_set.add(iii)

            for k in range(self.num_product):
                deg = str(round(len(i_set) * self.pw_list[k]))
                if deg in d_dict:
                    d_dict[deg].add((k, i))
                else:
                    d_dict[deg] = {(k, i)}

        return d_dict
