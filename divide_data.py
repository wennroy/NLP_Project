import csv

with open("Data/goemotions.csv", "w", encoding="UTF-8", newline='') as fw:
    csv_writer = csv.writer(fw)
    file_names = ['Data/goemotions_1.csv','Data/goemotions_2.csv','Data/goemotions_3.csv']
    total_num = 0
    total_num_with_unclear = 0

    for file_name in file_names:
        with open(file_name, 'r', encoding='UTF-8') as fr:
            csv_reader = csv.reader(fr)
            title = csv_reader.__next__()
            if file_name == 'Data/goemotions_1.csv':
                csv_writer.writerow(title)
            for data in csv_reader:
                total_num_with_unclear += 1
                if not data[8] == 'False':
                    continue
                total_num += 1
                csv_writer.writerow(data)
        print(f'Collected {total_num} samples, filtered from total {total_num_with_unclear} samples')

## Divided them into Train : Validate : Test = 8 : 1 : 1


end_of_train_num = round(total_num/10*8)
end_of_test_num = round(total_num/10*9)

with open("Data/goemotions.csv", "r", encoding="UTF-8") as fr:
    csv_reader = csv.reader(fr)
    title = csv_reader.__next__()
    count = 0
    with open("Data/goemotions_train.csv", "w", encoding="UTF-8", newline="") as fw:
        csv_writer = csv.writer(fw)
        csv_writer.writerow(title)
        for data in csv_reader:
            count += 1
            csv_writer.writerow(data)
            if count == end_of_train_num:
                break
        print(f'Found {count} samples as training data.')

    with open("Data/goemotions_dev.csv", "w", encoding="UTF-8", newline="") as fw:
        csv_writer = csv.writer(fw)
        csv_writer.writerow(title)
        for data in csv_reader:
            count += 1
            csv_writer.writerow(data)
            if count == end_of_test_num:
                break
        print(f'Found {count-end_of_train_num} samples as dev data.')

    with open("Data/goemotions_test.csv", "w", encoding="UTF-8", newline="") as fw:
        csv_writer = csv.writer(fw)
        csv_writer.writerow(title)
        for data in csv_reader:
            count += 1
            csv_writer.writerow(data)
            if count == end_of_test_num:
                break
        print(f'Found {count-end_of_test_num} samples as test data.')