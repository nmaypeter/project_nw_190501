from model import *

if __name__ == '__main__':
    dataset_seq = [1, 2, 3, 4, 5]
    prod_seq = [1, 2]
    cm_seq = [1, 2]
    wd_seq = [1, 2]
    for data_setting in dataset_seq:
        dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + 'email_Eu_core' * (data_setting == 3) + \
                       'WikiVote' * (data_setting == 4) + 'NetHEPT' * (data_setting == 5)
        for cm in cm_seq:
            cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
            for prod_setting in prod_seq:
                product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)

                Model(dataset_name, product_name, cascade_model).model_pmis()