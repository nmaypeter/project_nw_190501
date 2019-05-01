from SeedSelection_NaiveGreedy import *
from SeedSelection_NGAP import *
from SeedSelection_HighDegree import *
from SeedSelection_PMIS import *
from SeedSelection_Random import *


class Model:
    def __init__(self, dataset_name, product_name, cascade_model):
        self.dataset_name = dataset_name
        self.product_name = product_name
        self.cascade_model = cascade_model
        self.wd_seq = ['m50e25', 'm99e96']
        self.total_budget = 10
        self.wpiwp = bool(1)
        self.sample_number = 10
        self.ppp_seq = [2, 3]
        self.monte_carlo = 10

    def model_ng(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)

        seed_set_sequence = []
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', sample_count = ' + str(sample_count))
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssng_model.generateCelfHeap()
            mep_item = heap.heappop_max(celf_heap)
            mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    ep_g = 0.0
                    for _ in range(monte_carlo):
                        ep_g += diff_main.getSeedSetProfit(seed_set)
                    now_profit = round(ep_g / self.monte_carlo, 4)
                    now_budget = round(now_budget + sc, 4)
                else:
                    seed_set_t = copy.deepcopy(seed_set)
                    seed_set_t[mep_k_prod].add(mep_i_node)
                    ep_g = 0.0
                    for _ in range(self.monte_carlo):
                        ep_g += diff_model.getSeedSetProfit(seed_set_t)
                    ep_g = round(ep_g / self.monte_carlo, 4)
                    mg_g = round(ep_g - now_profit, 4)
                    flag_g = seed_set_length

                    if mg_g > 0:
                        celf_item_g = (mg_g, mep_k_prod, mep_i_node, flag_g)
                        heap.heappush_max(celf_heap, celf_item_g)

                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mng', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence)

    def model_ngr(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)

        seed_set_sequence = []
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', sample_count = ' + str(sample_count))
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssng_model.generateCelfHeapR()
            mep_item = heap.heappop_max(celf_heap)
            mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = heap.heappop_max(celf_heap)
                    mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    ep_g = 0.0
                    for _ in range(monte_carlo):
                        ep_g += diff_main.getSeedSetProfit(seed_set)
                    now_profit = round(ep_g / self.monte_carlo, 4)
                    now_budget = round(now_budget + sc, 4)
                else:
                    seed_set_t = copy.deepcopy(seed_set)
                    seed_set_t[mep_k_prod].add(mep_i_node)
                    ep_g = 0.0
                    for _ in range(self.monte_carlo):
                        ep_g += diff_model.getSeedSetProfit(seed_set_t)
                    ep_g = round(ep_g / self.monte_carlo, 4)
                    if sc == 0:
                        break
                    else:
                        mg_g = round(ep_g - now_profit, 4)
                        mg_ratio_g = round(mg_g / sc, 4)
                    flag_g = seed_set_length

                    if mg_ratio_g > 0:
                        celf_item_g = (mg_ratio_g, mep_k_prod, mep_i_node, flag_g)
                        heap.heappush_max(celf_heap, celf_item_g)

                mep_item = heap.heappop_max(celf_heap)
                mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mngr', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence)

    def model_ngsr(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)

        seed_set_sequence = []
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', sample_count = ' + str(sample_count))
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssng_model.generateCelfHeapR()
            mep_item = heap.heappop_max(celf_heap)
            mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = heap.heappop_max(celf_heap)
                    mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    ep_g = 0.0
                    for _ in range(monte_carlo):
                        ep_g += diff_main.getSeedSetProfit(seed_set)
                    now_profit = round(ep_g / self.monte_carlo, 4)
                    now_budget = round(now_budget + sc, 4)
                else:
                    seed_set_t = copy.deepcopy(seed_set)
                    seed_set_t[mep_k_prod].add(mep_i_node)
                    ep_g = 0.0
                    for _ in range(self.monte_carlo):
                        ep_g += diff_model.getSeedSetProfit(seed_set_t)
                    ep_g = round(ep_g / self.monte_carlo, 4)
                    if (now_budget + sc) == 0:
                        break
                    else:
                        mg_g = round(ep_g - now_profit, 4)
                        mg_seed_ratio_g = round(mg_g / (now_budget + sc), 4)
                    flag_g = seed_set_length

                    if mg_seed_ratio_g > 0:
                        celf_item_g = (mg_seed_ratio_g, mep_k_prod, mep_i_node, flag_g)
                        heap.heappush_max(celf_heap, celf_item_g)

                mep_item = heap.heappop_max(celf_heap)
                mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mngsr', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence)

    def model_ngap(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)

        seed_set_sequence = []
        ssngap_model = SeedSelectionNGAP(graph_dict, seed_cost_dict, product_list)
        diffap_model = DiffusionAccProb(graph_dict, seed_cost_dict, product_list)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', sample_count = ' + str(sample_count))
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            celf_heap = ssngap_model.generateCelfHeap()
            mep_item = heap.heappop_max(celf_heap)
            mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_profit = round(now_profit + mep_mg, 4)
                    now_budget = round(now_budget + sc, 4)
                    expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_mg, 4)
                else:
                    seed_set_k = copy.deepcopy(seed_set[mep_k_prod])
                    seed_set_k.add(mep_i_node)
                    s_dict = {}
                    for s in seed_set_k:
                        i_dict = diffap_model.buildNodeDict(seed_set_k, s, 1)
                        for ii in i_dict:
                            if ii not in s_dict:
                                s_dict[ii] = i_dict[ii]
                            else:
                                s_dict[ii] += i_dict[ii]
                    expected_inf = getExpectedInf(s_dict)
                    ep_g = round(expected_inf * product_list[mep_k_prod][0], 4)
                    mg_g = round(ep_g - expected_profit_k[mep_k_prod], 4)
                    flag_g = seed_set_length

                    if mg_g > 0:
                        celf_item_g = (mg_g, mep_k_prod, mep_i_node, flag_g)
                        heap.heappush_max(celf_heap, celf_item_g)

                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mngap', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence)

    def model_ngapr(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)

        seed_set_sequence = []
        ssngap_model = SeedSelectionNGAP(graph_dict, seed_cost_dict, product_list)
        diffap_model = DiffusionAccProb(graph_dict, seed_cost_dict, product_list)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', sample_count = ' + str(sample_count))
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            celf_heap = ssngap_model.generateCelfHeapR()
            mep_item = heap.heappop_max(celf_heap)
            mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = heap.heappop_max(celf_heap)
                    mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_profit = round(now_profit + mep_ratio * sc, 4)
                    now_budget = round(now_budget + sc, 4)
                    expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_ratio * sc, 4)
                else:
                    seed_set_k = copy.deepcopy(seed_set[mep_k_prod])
                    seed_set_k.add(mep_i_node)
                    s_dict = {}
                    for s in seed_set_k:
                        i_dict = diffap_model.buildNodeDict(seed_set_k, s, 1)
                        for ii in i_dict:
                            if ii not in s_dict:
                                s_dict[ii] = i_dict[ii]
                            else:
                                s_dict[ii] += i_dict[ii]
                    expected_inf = getExpectedInf(s_dict)
                    ep_g = round(expected_inf * product_list[mep_k_prod][0], 4)
                    if sc == 0:
                        break
                    else:
                        mg_g = round(ep_g - expected_profit_k[mep_k_prod], 4)
                        mg_ratio_g = round(mg_g / sc, 4)
                    flag_g = seed_set_length

                    if mg_ratio_g > 0:
                        celf_item_g = (mg_ratio_g, mep_k_prod, mep_i_node, flag_g)
                        heap.heappush_max(celf_heap, celf_item_g)

                mep_item = heap.heappop_max(celf_heap)
                mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mngapr', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence)

    def model_ngapsr(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)

        seed_set_sequence = []
        ssngap_model = SeedSelectionNGAP(graph_dict, seed_cost_dict, product_list)
        diffap_model = DiffusionAccProb(graph_dict, seed_cost_dict, product_list)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', sample_count = ' + str(sample_count))
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            celf_heap = ssngap_model.generateCelfHeapR()
            mep_item = heap.heappop_max(celf_heap)
            mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = heap.heappop_max(celf_heap)
                    mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_profit = round(now_profit + mep_seed_ratio * (now_budget + sc), 4)
                    now_budget = round(now_budget + sc, 4)
                    expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_seed_ratio * now_budget, 4)
                else:
                    seed_set_k = copy.deepcopy(seed_set[mep_k_prod])
                    seed_set_k.add(mep_i_node)
                    s_dict = {}
                    for s in seed_set_k:
                        i_dict = diffap_model.buildNodeDict(seed_set_k, s, 1)
                        for ii in i_dict:
                            if ii not in s_dict:
                                s_dict[ii] = i_dict[ii]
                            else:
                                s_dict[ii] += i_dict[ii]
                    expected_inf = getExpectedInf(s_dict)
                    ep_g = round(expected_inf * product_list[mep_k_prod][0], 4)
                    if (now_budget + sc) == 0:
                        break
                    else:
                        mg_g = round(ep_g - expected_profit_k[mep_k_prod], 4)
                        mg_seed_ratio_g = round(mg_g / (now_budget + sc), 4)
                    flag_g = seed_set_length

                    if mg_seed_ratio_g > 0:
                        celf_item_g = (mg_seed_ratio_g, mep_k_prod, mep_i_node, flag_g)
                        heap.heappush_max(celf_heap, celf_item_g)

                mep_item = heap.heappop_max(celf_heap)
                mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mngapsr', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence)

    def model_hd(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)

        seed_set_sequence = []
        sshd_model = SeedSelectionHD(graph_dict, seed_cost_dict, product_list)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', sample_count = ' + str(sample_count))
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            degree_dict = sshd_model.constructDegreeDict(self.dataset_name)
            mep_item = selectDegreeSeed(degree_dict)
            mep_k_prod, mep_i_node = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = selectDegreeSeed(degree_dict)
                    mep_k_prod, mep_i_node = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                seed_set[mep_k_prod].add(mep_i_node)
                now_budget = round(now_budget + sc, 4)

                mep_item = selectDegreeSeed(degree_dict)
                mep_k_prod, mep_i_node = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mhd', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence)

    def model_hed(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)

        seed_set_sequence = []
        sshd_model = SeedSelectionHD(graph_dict, seed_cost_dict, product_list)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', sample_count = ' + str(sample_count))
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            degree_dict = sshd_model.constructExpendDegreeDict()
            mep_item = selectDegreeSeed(degree_dict)
            mep_k_prod, mep_i_node = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = selectDegreeSeed(degree_dict)
                    mep_k_prod, mep_i_node = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                seed_set[mep_k_prod].add(mep_i_node)
                now_budget = round(now_budget + sc, 4)

                mep_item = selectDegreeSeed(degree_dict)
                mep_k_prod, mep_i_node = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mhed', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence)

    def model_pmis(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)

        seed_set_sequence = []
        sspmis_model = SeedSelectionPMIS(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', sample_count = ' + str(sample_count))
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            s_matrix, c_matrix = [[set() for _ in range(num_product)]], [0.0]
            celf_heap = sspmis_model.generateCelfHeap()
            mep_item = heap.heappop_max(celf_heap)
            mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    ep_g = 0.0
                    for _ in range(monte_carlo):
                        ep_g += diff_main.getSeedSetProfit(seed_set)
                    now_profit = round(ep_g / self.monte_carlo, 4)
                    now_budget = round(now_budget + sc, 4)
                    s_matrix.append(copy.deepcopy(seed_set))
                    c_matrix.append(now_budget)
                else:
                    seed_set_t = copy.deepcopy(seed_set)
                    seed_set_t[mep_k_prod].add(mep_i_node)
                    ep_g = 0.0
                    for _ in range(self.monte_carlo):
                        ep_g += diff_model.getSeedSetProfit(seed_set_t)
                    ep_g = round(ep_g / self.monte_carlo, 4)
                    mg_g = round(ep_g - now_profit, 4)
                    flag_g = seed_set_length

                    if mg_g > 0:
                        celf_item_g = (mg_g, mep_k_prod, mep_i_node, flag_g)
                        heap.heappush_max(celf_heap, celf_item_g)

                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                s_matrix = [copy.deepcopy(s_matrix) for _ in range(num_product)]
                for kk in range(1, num_product):
                    for kk_item in s_matrix[kk]:
                        kk_item[0], kk_item[kk] = kk_item[kk], kk_item[0]
                c_matrix = [c_matrix for _ in range(num_product)]

                mep_result = [0.0, [set() for _ in range(num_product)]]
                ### bud_index: (list) the using budget index for products
                ### bud_bound_index: (list) the bound budget index for products
                bud_index, bud_bound_index = [len(kk) - 1 for kk in c_matrix], [0 for _ in range(num_product)]
                ### temp_bound_index: (list) the bound to exclude the impossible budget combination s.t. the k-budget is smaller than the temp bound
                temp_bound_index = [0 for _ in range(num_product)]

                while not operator.eq(bud_index, bud_bound_index):
                    ### bud_pmis: (float) the budget in this pmis execution
                    bud_pmis = sum(c_matrix[kk][bud_index[kk]] for kk in range(num_product))

                    if bud_pmis <= self.total_budget:
                        temp_bound_flag = 0
                        for kk in range(num_product):
                            if temp_bound_index[kk] >= bud_index[kk]:
                                temp_bound_flag += 1

                        if temp_bound_flag != num_product:
                            temp_bound_index = copy.deepcopy(bud_index)
                            # -- pmis execution --
                            seed_set_pmis = [set() for _ in range(num_product)]
                            for kk in range(num_product):
                                seed_set_pmis[kk] = s_matrix[kk][bud_index[kk]][kk]

                            pro_acc = 0.0
                            for _ in range(self.monte_carlo):
                                pro_acc += diff_model.getSeedSetProfit(seed_set_pmis)
                            pro_acc = round(pro_acc / self.monte_carlo, 4)

                            if pro_acc > mep_result[0]:
                                mep_result = [pro_acc, seed_set_pmis]

                    pointer = num_product - 1
                    while bud_index[pointer] == bud_bound_index[pointer]:
                        bud_index[pointer] = len(c_matrix[pointer]) - 1
                        pointer -= 1
                    bud_index[pointer] -= 1
                seed_set = mep_result[1]

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mpmis', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence)

    def model_r(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)

        seed_set_sequence = []
        ssr_model = SeedSelectionRandom(graph_dict, seed_cost_dict, product_list)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', sample_count = ' + str(sample_count))
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            random_node_set = ssr_model.constructRandomNodeSet()
            mep_item = selectRandomSeed(random_node_set)
            mep_k_prod, mep_i_node = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = selectRandomSeed(random_node_set)
                    mep_k_prod, mep_i_node = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                seed_set[mep_k_prod].add(mep_i_node)
                now_budget = round(now_budget + sc, 4)

                mep_item = selectRandomSeed(random_node_set)
                mep_k_prod, mep_i_node = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mr', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence)


class ModelPW:
    def __init__(self, dataset_name, product_name, cascade_model, wallet_distribution_type):
        self.dataset_name = dataset_name
        self.product_name = product_name
        self.cascade_model = cascade_model
        self.wallet_distribution_type = wallet_distribution_type
        self.total_budget = 10
        self.wpiwp = bool(1)
        self.sample_number = 10
        self.ppp_seq = [2, 3]

    def model_ngpw(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)

        seed_set_sequence = []
        ssngpw_model = SeedSelectionNGPW(graph_dict, seed_cost_dict, product_list, product_weight_list, self.monte_carlo)
        diffpw_model = DiffusionPW(graph_dict, seed_cost_dict, product_list, product_weight_list, self.monte_carlo)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', wd = ' + str(self.wallet_distribution_type) + ', sample_count = ' + str(sample_count))
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssngpw_model.generateCelfHeap()
            mep_item = heap.heappop_max(celf_heap)
            mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    ep_g = 0.0
                    for _ in range(monte_carlo):
                        ep_g += diff_main.getSeedSetProfit(seed_set)
                    now_profit = round(ep_g / self.monte_carlo, 4)
                    now_budget = round(now_budget + sc, 4)
                else:
                    seed_set_t = copy.deepcopy(seed_set)
                    seed_set_t[mep_k_prod].add(mep_i_node)
                    ep_g = 0.0
                    for _ in range(self.monte_carlo):
                        ep_g += diffpw_model.getSeedSetProfit(seed_set_t)
                    ep_g = round(ep_g / self.monte_carlo, 4)
                    mg_g = round(ep_g - now_profit, 4)
                    flag_g = seed_set_length

                    if mg_g > 0:
                        celf_item_g = (mg_g, mep_k_prod, mep_i_node, flag_g)
                        heap.heappush_max(celf_heap, celf_item_g)

                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mngpw', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for ppp in self.ppp_seq:
            eva_model.evaluate(self.wallet_distribution_type, ppp, seed_set_sequence)

    def model_ngrpw(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)

        seed_set_sequence = []
        ssngpw_model = SeedSelectionNGPW(graph_dict, seed_cost_dict, product_list, product_weight_list, self.monte_carlo)
        diffpw_model = DiffusionPW(graph_dict, seed_cost_dict, product_list, product_weight_list, self.monte_carlo)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', wd = ' + str(self.wallet_distribution_type) + ', sample_count = ' + str(sample_count))
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssngpw_model.generateCelfHeapR()
            mep_item = heap.heappop_max(celf_heap)
            mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = heap.heappop_max(celf_heap)
                    mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    ep_g = 0.0
                    for _ in range(monte_carlo):
                        ep_g += diff_main.getSeedSetProfit(seed_set)
                    now_profit = round(ep_g / self.monte_carlo, 4)
                    now_budget = round(now_budget + sc, 4)
                else:
                    seed_set_t = copy.deepcopy(seed_set)
                    seed_set_t[mep_k_prod].add(mep_i_node)
                    ep_g = 0.0
                    for _ in range(self.monte_carlo):
                        ep_g += diffpw_model.getSeedSetProfit(seed_set_t)
                    ep_g = round(ep_g / self.monte_carlo, 4)
                    if sc == 0:
                        break
                    else:
                        mg_g = round(ep_g - now_profit, 4)
                        mg_ratio_g = round(mg_g / sc, 4)
                    flag_g = seed_set_length

                    if mg_ratio_g > 0:
                        celf_item_g = (mg_ratio_g, mep_k_prod, mep_i_node, flag_g)
                        heap.heappush_max(celf_heap, celf_item_g)

                mep_item = heap.heappop_max(celf_heap)
                mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mngrpw', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for ppp in self.ppp_seq:
            eva_model.evaluate(self.wallet_distribution_type, ppp, seed_set_sequence)

    def model_ngsrpw(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)

        seed_set_sequence = []
        ssngpw_model = SeedSelectionNGPW(graph_dict, seed_cost_dict, product_list, product_weight_list, self.monte_carlo)
        diffpw_model = DiffusionPW(graph_dict, seed_cost_dict, product_list, product_weight_list, self.monte_carlo)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', wd = ' + str(self.wallet_distribution_type) + ', sample_count = ' + str(sample_count))
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssngpw_model.generateCelfHeapR()
            mep_item = heap.heappop_max(celf_heap)
            mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = heap.heappop_max(celf_heap)
                    mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    ep_g = 0.0
                    for _ in range(monte_carlo):
                        ep_g += diff_main.getSeedSetProfit(seed_set)
                    now_profit = round(ep_g / self.monte_carlo, 4)
                    now_budget = round(now_budget + sc, 4)
                else:
                    seed_set_t = copy.deepcopy(seed_set)
                    seed_set_t[mep_k_prod].add(mep_i_node)
                    ep_g = 0.0
                    for _ in range(self.monte_carlo):
                        ep_g += diffpw_model.getSeedSetProfit(seed_set_t)
                    ep_g = round(ep_g / self.monte_carlo, 4)
                    if (now_budget + sc) == 0:
                        break
                    else:
                        mg_g = round(ep_g - now_profit, 4)
                        mg_seed_ratio_g = round(mg_g / (now_budget + sc), 4)
                    flag_g = seed_set_length

                    if mg_seed_ratio_g > 0:
                        celf_item_g = (mg_seed_ratio_g, mep_k_prod, mep_i_node, flag_g)
                        heap.heappush_max(celf_heap, celf_item_g)

                mep_item = heap.heappop_max(celf_heap)
                mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mngrpw', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for ppp in self.ppp_seq:
            eva_model.evaluate(self.wallet_distribution_type, ppp, seed_set_sequence)

    def model_ngappw(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)

        seed_set_sequence = []
        ssngappw_model = SeedSelectionNGAPPW(graph_dict, seed_cost_dict, product_list, product_weight_list)
        diffap_model = DiffusionAccProb(graph_dict, seed_cost_dict, product_list)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', wd = ' + str(self.wallet_distribution_type) + ', sample_count = ' + str(sample_count))
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            celf_heap = ssngappw_model.generateCelfHeap()
            mep_item = heap.heappop_max(celf_heap)
            mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_profit = round(now_profit + mep_mg, 4)
                    now_budget = round(now_budget + sc, 4)
                    expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_mg, 4)
                else:
                    seed_set_k = copy.deepcopy(seed_set[mep_k_prod])
                    seed_set_k.add(mep_i_node)
                    s_dict = {}
                    for s in seed_set_k:
                        i_dict = diffap_model.buildNodeDict(seed_set_k, s, 1)
                        for ii in i_dict:
                            if ii not in s_dict:
                                s_dict[ii] = i_dict[ii]
                            else:
                                s_dict[ii] += i_dict[ii]
                    expected_inf = getExpectedInf(s_dict)
                    ep_g = round(expected_inf * product_list[mep_k_prod][0] * product_weight_list[mep_k_prod], 4)
                    mg_g = round(ep_g - expected_profit_k[mep_k_prod], 4)
                    flag_g = seed_set_length

                    if mg_g > 0:
                        celf_item_g = (mg_g, mep_k_prod, mep_i_node, flag_g)
                        heap.heappush_max(celf_heap, celf_item_g)

                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mngappw', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for ppp in self.ppp_seq:
            eva_model.evaluate(self.wallet_distribution_type, ppp, seed_set_sequence)

    def model_ngaprpw(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)

        seed_set_sequence = []
        ssngappw_model = SeedSelectionNGAPPW(graph_dict, seed_cost_dict, product_list, product_weight_list)
        diffap_model = DiffusionAccProb(graph_dict, seed_cost_dict, product_list)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', wd = ' + str(self.wallet_distribution_type) + ', sample_count = ' + str(sample_count))
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            celf_heap = ssngappw_model.generateCelfHeapR()
            mep_item = heap.heappop_max(celf_heap)
            mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = heap.heappop_max(celf_heap)
                    mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_profit = round(now_profit + mep_ratio * sc, 4)
                    now_budget = round(now_budget + sc, 4)
                    expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_ratio * sc, 4)
                else:
                    seed_set_k = copy.deepcopy(seed_set[mep_k_prod])
                    seed_set_k.add(mep_i_node)
                    s_dict = {}
                    for s in seed_set_k:
                        i_dict = diffap_model.buildNodeDict(seed_set_k, s, 1)
                        for ii in i_dict:
                            if ii not in s_dict:
                                s_dict[ii] = i_dict[ii]
                            else:
                                s_dict[ii] += i_dict[ii]
                    expected_inf = getExpectedInf(s_dict)
                    ep_g = round(expected_inf * product_list[mep_k_prod][0] * product_weight_list[mep_k_prod], 4)
                    if sc == 0:
                        break
                    else:
                        mg_g = round(ep_g - expected_profit_k[mep_k_prod], 4)
                        mg_ratio_g = round(mg_g / sc, 4)
                    flag_g = seed_set_length

                    if mg_ratio_g > 0:
                        celf_item_g = (mg_ratio_g, mep_k_prod, mep_i_node, flag_g)
                        heap.heappush_max(celf_heap, celf_item_g)

                mep_item = heap.heappop_max(celf_heap)
                mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mngaprpw', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for ppp in self.ppp_seq:
            eva_model.evaluate(self.wallet_distribution_type, ppp, seed_set_sequence)

    def model_ngapsrpw(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)

        seed_set_sequence = []
        ssngappw_model = SeedSelectionNGAPPW(graph_dict, seed_cost_dict, product_list, product_weight_list)
        diffap_model = DiffusionAccProb(graph_dict, seed_cost_dict, product_list)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', wd = ' + str(self.wallet_distribution_type) + ', sample_count = ' + str(sample_count))
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            celf_heap = ssngappw_model.generateCelfHeapR()
            mep_item = heap.heappop_max(celf_heap)
            mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = heap.heappop_max(celf_heap)
                    mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_profit = round(now_profit + mep_seed_ratio * (now_budget + sc), 4)
                    now_budget = round(now_budget + sc, 4)
                    expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_seed_ratio * now_budget, 4)
                else:
                    seed_set_k = copy.deepcopy(seed_set[mep_k_prod])
                    seed_set_k.add(mep_i_node)
                    s_dict = {}
                    for s in seed_set_k:
                        i_dict = diffap_model.buildNodeDict(seed_set_k, s, 1)
                        for ii in i_dict:
                            if ii not in s_dict:
                                s_dict[ii] = i_dict[ii]
                            else:
                                s_dict[ii] += i_dict[ii]
                    expected_inf = getExpectedInf(s_dict)
                    ep_g = round(expected_inf * product_list[mep_k_prod][0] * product_weight_list[mep_k_prod], 4)
                    if (now_budget + sc) == 0:
                        break
                    else:
                        mg_g = round(ep_g - expected_profit_k[mep_k_prod], 4)
                        mg_seed_ratio_g = round(mg_g / (now_budget + sc), 4)
                    flag_g = seed_set_length

                    if mg_seed_ratio_g > 0:
                        celf_item_g = (mg_seed_ratio_g, mep_k_prod, mep_i_node, flag_g)
                        heap.heappush_max(celf_heap, celf_item_g)

                mep_item = heap.heappop_max(celf_heap)
                mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mngapsrpw', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for ppp in self.ppp_seq:
            eva_model.evaluate(self.wallet_distribution_type, ppp, seed_set_sequence)

    def model_hdpw(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)

        seed_set_sequence = []
        sshdpw_model = SeedSelectionHDPW(graph_dict, seed_cost_dict, product_list, product_weight_list)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', wd = ' + str(self.wallet_distribution_type) + ', sample_count = ' + str(sample_count))
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            degree_dict = sshdpw_model.constructDegreeDict(self.dataset_name)
            mep_item = selectDegreeSeed(degree_dict)
            mep_k_prod, mep_i_node = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = selectDegreeSeed(degree_dict)
                    mep_k_prod, mep_i_node = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                seed_set[mep_k_prod].add(mep_i_node)
                now_budget = round(now_budget + sc, 4)

                mep_item = selectDegreeSeed(degree_dict)
                mep_k_prod, mep_i_node = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mhdpw', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for ppp in self.ppp_seq:
            eva_model.evaluate(self.wallet_distribution_type, ppp, seed_set_sequence)

    def model_hedpw(self):
        ss_strat_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)

        seed_set_sequence = []
        sshdpw_model = SeedSelectionHDPW(graph_dict, seed_cost_dict, product_list, product_weight_list)
        for sample_count in range(self.sample_number):
            print('@ seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', wd = ' + str(self.wallet_distribution_type) + ', sample_count = ' + str(sample_count))
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            degree_dict = sshdpw_model.constructExpendDegreeDict()
            mep_item = selectDegreeSeed(degree_dict)
            mep_k_prod, mep_i_node = mep_item

            while now_budget < self.total_budget and mep_i_node != '-1':
                sc = seed_cost_dict[mep_i_node]
                if round(now_budget + sc, 4) > self.total_budget:
                    mep_item = selectDegreeSeed(degree_dict)
                    mep_k_prod, mep_i_node = mep_item
                    if mep_i_node == '-1':
                        break
                    continue

                seed_set[mep_k_prod].add(mep_i_node)
                now_budget = round(now_budget + sc, 4)

                mep_item = selectDegreeSeed(degree_dict)
                mep_k_prod, mep_i_node = mep_item

            print('ss_time = ' + str(round(time.time() - ss_strat_time, 2)) + 'sec')
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_strat_time, 2)
        eva_model = EvaluationM('mhedpw', self.dataset_name, self.product_name, self.cascade_model, ss_time)
        for ppp in self.ppp_seq:
            eva_model.evaluate(self.wallet_distribution_type, ppp, seed_set_sequence)