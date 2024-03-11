### FOR amie_rules_all_train.txt RULES
start_filepath = ["train_processed.nl","test_processed.nl","dev_processed.nl"]
stop_filepath = ["train_amie.txt","test_amie.txt","dev_amie.txt"]

for e, i in enumerate(start_filepath):
    orig_file = open(i, "r")
    orig_facts = orig_file.readlines()

    with open(stop_filepath[e], "w") as f:
        for fact in orig_facts:
            fact_lst = fact.split()
            AMIE_fact = fact_lst[1][1:-1] + " " + fact_lst[0] + " " + fact_lst[2][:-2]
            f.write(AMIE_fact+"\n")
    f.close()


