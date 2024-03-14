### FOR UniKER RULES
start_filepath = "old_maybe_to_clean/ordered_amie_rules.text"
stop_filepath = "rules.txt"


UniKER_file = open(start_filepath, "r")
UniKER_rules = UniKER_file.readlines()
with open(stop_filepath, "w") as f:
    for idx, rule in enumerate(UniKER_rules):
        R2N_rule = "r"+str(idx) + ":"
        rule_list = rule.split()
        rule_lenght = len(rule_list)
        if rule_lenght == 7+7:
            R2N_rule += rule_list[9] + ":"
            R2N_rule += rule_list[1] + "(" + rule_list[0][1:] + "," + rule_list[2][1:] + ")" + " -> "
            R2N_rule += rule_list[5] + "(" + rule_list[4][1:] + "," + rule_list[6][1:] + ")"
        if rule_lenght == 10+7:
            R2N_rule += rule_list[12] + ":"
            R2N_rule += rule_list[1] + "(" + rule_list[0][1:] + "," + rule_list[2][1:] + ")" + ", "
            R2N_rule += rule_list[4] + "(" + rule_list[3][1:] + "," + rule_list[5][1:] + ")" + " -> "
            R2N_rule += rule_list[8] + "(" + rule_list[7][1:] + "," + rule_list[9][1:] + ")"
        f.write(R2N_rule+"\n")
    f.close()


