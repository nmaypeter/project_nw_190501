from Diffusion import *
import heap


class SeedSelectionNGAP:
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

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        i_anc_dict = {}
        celf_heap = [(0.0, -1, '-1', 0)]
        mep = (0.0, {})
        diffap_ss = DiffusionAccProb(self.graph_dict, self.seed_cost_dict, self.product_list)

        for i in self.graph_dict:
            i_dict = diffap_ss.buildNodeDict({i}, i, 1, set())
            i_anc_dict[i] = i_dict
            ei = getExpectedInf(i_dict)

            if ei > 0:
                for k in range(self.num_product):
                    mg = round(ei * self.product_list[k][0], 4)
                    if mg > mep[0]:
                        mep = (mg, i_dict)
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap, i_anc_dict, mep

    def generateCelfHeapR(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg_ratio, k_prod, i_node, flag)
        i_anc_dict = {}
        celf_heap = [(0.0, -1, '-1', 0)]
        mep = (0.0, {})
        diffap_ss = DiffusionAccProb(self.graph_dict, self.seed_cost_dict, self.product_list)

        for i in self.graph_dict:
            i_dict = diffap_ss.buildNodeDict({i}, i, 1, set())
            i_anc_dict[i] = i_dict
            ei = getExpectedInf(i_dict)

            if ei > 0:
                for k in range(self.num_product):
                    if self.seed_cost_dict[i] == 0:
                        break
                    else:
                        mg_ratio = round(ei * self.product_list[k][0] / self.seed_cost_dict[i], 4)
                    if mg_ratio > mep[0]:
                        mep = (mg_ratio, i_dict)
                    celf_item = (mg_ratio, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap, i_anc_dict, mep


class SeedSelectionNGAPPW:
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

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        i_anc_dict = {}
        celf_heap = [(0.0, -1, '-1', 0)]
        mep = (0.0, {})
        diffap_ss = DiffusionAccProb(self.graph_dict, self.seed_cost_dict, self.product_list)

        for i in self.graph_dict:
            i_dict = diffap_ss.buildNodeDict({i}, i, 1, set())
            i_anc_dict[i] = i_dict
            ei = getExpectedInf(i_dict)

            if ei > 0:
                for k in range(self.num_product):
                    mg = round(ei * self.product_list[k][0] * self.pw_list[k], 4)
                    if mg > mep[0]:
                        mep = (mg, i_dict)
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap, i_anc_dict, mep

    def generateCelfHeapR(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg_ratio, k_prod, i_node, flag)
        i_anc_dict = {}
        celf_heap = [(0.0, -1, '-1', 0)]
        mep = (0.0, {})
        diffap_ss = DiffusionAccProb(self.graph_dict, self.seed_cost_dict, self.product_list)

        for i in self.graph_dict:
            i_dict = diffap_ss.buildNodeDict({i}, i, 1, set())
            i_anc_dict[i] = i_dict
            ei = getExpectedInf(i_dict)

            if ei > 0:
                for k in range(self.num_product):
                    if self.seed_cost_dict[i] == 0:
                        break
                    else:
                        mg_ratio = round(ei * self.product_list[k][0] * self.pw_list[k] / self.seed_cost_dict[i], 4)
                    if mg_ratio > mep[0]:
                        mep = (mg_ratio, i_dict)
                    celf_item = (mg_ratio, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap, i_anc_dict, mep