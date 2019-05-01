from Diffusion import *


def selectRandomSeed(rn_set):
    # -- select a seed for a random product randomly --
    mep = (-1, '-1')
    if len(rn_set) != 0:
        mep = choice(list(rn_set))
        rn_set.remove(mep)

    return mep


class SeedSelectionRandom:
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

    def constructRandomNodeSet(self):
        rn_set = set()
        for k in range(self.num_product):
            for i in self.graph_dict:
                rn_set.add((k, i))

        return rn_set