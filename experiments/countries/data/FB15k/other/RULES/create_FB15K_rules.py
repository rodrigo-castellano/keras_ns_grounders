gen_rules = "EGNN"

if gen_rules == "RUGE":
    ### FOR RUGE RULES
    start_filepath_ruge = "rules_RUGE.txt"
    stop_filepath_ruge = "rules_RUGE2R2N.txt"

    RUGE_file = open(start_filepath_ruge, "r")
    RUGE_rules = RUGE_file.readlines()
    with open(stop_filepath_ruge, "w") as f:
        for idx, rule in enumerate(RUGE_rules):
            R2N_rule = "r"+str(idx)+":0:"
            rule_list = rule.split()
            rule_lenght = len(rule_list)
            i=0
            body = True
            while i<rule_lenght-1:
                # "=>" split body from head
                if rule_list[i] == "=>":
                    R2N_rule += " -> "
                    i += 1
                    body = False
                    continue
                if body == True and i>0:
                    R2N_rule += ","
                R2N_rule += rule_list[i + 1] + "(" + rule_list[i][1] + "," + rule_list[i + 2][1] + ")"
                i += 3

            f.write(R2N_rule+"\n")
        f.close()
else:

    ### FOR ExpressGNN RULES
    start_filepath_EGNN = "rules_EGNN.txt"
    stop_filepath_EGNN = "rules_EGNN2R2N.txt"

    EGNN_file = open(start_filepath_EGNN, "r")
    EGNN_rules = EGNN_file.readlines()
    with open(stop_filepath_EGNN, "w") as f:
        for idx, rule in enumerate(EGNN_rules):
            R2N_rule = "r"+str(idx)+":0:"
            rule_list = rule.split()
            rule_lenght = len(rule_list)
            i=0
            while i<rule_lenght:
                # NO "!" split body from head
                if rule_list[i][0] == "!":
                    if i>0:
                        R2N_rule += ","
                    R2N_rule += rule_list[i][1:]
                else:
                    R2N_rule += " -> " + rule_list[i]
                i += 2
            f.write(R2N_rule+"\n")
        f.close()


