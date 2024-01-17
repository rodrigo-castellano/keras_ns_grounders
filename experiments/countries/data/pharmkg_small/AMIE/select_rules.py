starting_rules = open("rules_train2test_raw.txt", "r")
rules = starting_rules.readlines()

RULE_LIST=[]

with open("../rules_train2test.txt", "w") as f:
    for rule in rules:
        rule_list = rule.split(sep="\t")
        rule_list[1] = float(rule_list[1])
        rule_list.reverse()
        RULE_LIST.append(rule_list)
    RULE_LIST_ordered = sorted(RULE_LIST, reverse=True)

    for c, rule in enumerate(RULE_LIST_ordered):
        R2N_rule = "r"+str(c)+":"
        R2N_rule += str(rule[0]/len(RULE_LIST_ordered))+":"
        R2N_rule += rule[1]
        f.write(R2N_rule+"\n")
    f.close()
