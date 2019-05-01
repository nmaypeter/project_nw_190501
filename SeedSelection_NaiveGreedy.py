from Diffusion import *
import heap


class SeedSelectionNG:
    def __init__(self, g_dict, s_c_dict, prod_list, monte):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### monte: (int) monte carlo times
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.monte = monte

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        celf_heap = [(0.0, -1, '-1', 0)]

        diff_ss = Diffusion(self.graph_dict, self.seed_cost_dict, self.product_list, self.monte)

        for i in set(self.graph_dict.keys()):
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = 0.0
            for _ in range(self.monte):
                ep += diff_ss.getSeedSetProfit(s_set)
            ep = round(ep / self.monte, 4)

            if ep > 0:
                for k in range(self.num_product):
                    mg = round(ep * self.product_list[k][0] / self.product_list[0][0], 4)
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap

    def generateCelfHeapR(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg_ratio, k_prod, i_node, flag)
        celf_heap = [(0.0, -1, '-1', 0)]

        diff_ss = Diffusion(self.graph_dict, self.seed_cost_dict, self.product_list, self.monte)

        for i in set(self.graph_dict.keys()):
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = 0.0
            for _ in range(self.monte):
                ep += diff_ss.getSeedSetProfit(s_set)
            ep = round(ep / self.monte, 4)

            if ep > 0:
                for k in range(self.num_product):
                    if self.seed_cost_dict[i] == 0:
                        break
                    else:
                        mg = round(ep * self.product_list[k][0] / self.product_list[0][0], 4)
                        mg_ratio = round(mg / self.seed_cost_dict[i], 4)
                    celf_item = (mg_ratio, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap


class SeedSelectionNGPW:
    def __init__(self, g_dict, s_c_dict, prod_list, pw_list, monte):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### pw_list: (list) the product weight list
        ### monte: (int) monte carlo times
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.pw_list = pw_list
        self.monte = monte

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        celf_heap = [(0.0, -1, '-1', 0)]

        diffpw_ss = DiffusionPW(self.graph_dict, self.seed_cost_dict, self.product_list, self.pw_list, self.monte)

        for i in set(self.graph_dict.keys()):
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = 0.0
            for _ in range(self.monte):
                ep += diffpw_ss.getSeedSetProfit(s_set)
            ep = round(ep / self.monte, 4)

            if ep > 0:
                for k in range(self.num_product):
                    mg = round(ep * self.product_list[k][0] * self.pw_list[k] / (self.product_list[0][0] * self.pw_list[0]), 4)
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap

    def generateCelfHeapR(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg_ratio, k_prod, i_node, flag)
        celf_heap = [(0.0, -1, '-1', 0)]

        diffpw_ss = DiffusionPW(self.graph_dict, self.seed_cost_dict, self.product_list, self.pw_list, self.monte)

        for i in set(self.graph_dict.keys()):
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = 0.0
            for _ in range(self.monte):
                ep += diffpw_ss.getSeedSetProfit(s_set)
            ep = round(ep / self.monte, 4)

            if ep > 0:
                for k in range(self.num_product):
                    if self.seed_cost_dict[i] == 0:
                        break
                    else:
                        mg = round(ep * self.product_list[k][0] * self.pw_list[k] / (self.product_list[0][0] * self.pw_list[0]), 4)
                        mg_ratio = round(mg / self.seed_cost_dict[i], 4)
                    celf_item = (mg_ratio, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap