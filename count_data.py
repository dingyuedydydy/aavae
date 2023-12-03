Dataset_path = 'beauty'
user_max = 0
item_max = 0
seq_len = 0
prev_user = 0
count = 1
records = 0
print('Loading data...')
fin = open('datasets/' + Dataset_path + '.txt', 'rt')
for line in fin.readlines():
    records += 1
    user_id, item_id = line.split(' ')
    user_id = int(user_id) - 1
    item_id = int(item_id)
    if user_id > user_max:
        user_max = user_id
    if item_id > item_max:
        item_max = item_id
    if user_id == prev_user:
        count = count + 1
    else:
        prev_user = user_id
        if count > seq_len:
            seq_len = count
        count = 1

print(records, user_max + 1, item_max, seq_len)
