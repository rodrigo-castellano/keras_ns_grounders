with open('../train.txt') as f:
    lines = f.readlines()
    lines = [line.replace('\t', ' ') for line in lines]
file = open('train_amieKB_to_process.txt', 'w+')
for line in lines:
    file.write(line)

print('end')
