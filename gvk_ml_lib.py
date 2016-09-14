# importing pandas
import pandas as pd

# importing numpy
import numpy as np


def read_csv_file(filename=''):
    complete_data = pd.read_csv(filename)
    print 'file read ...'
    return complete_data


def get_data_header(data):
    data_header = list(data.columns.values)
    return data_header


def get_class_of_data(data):
    class_of_data = data.iloc[0:, -1]
    return class_of_data


def get_max_col_value(column):
    return column.max(axis=0)


def get_min_col_value(column):
    return column.min(axis=0)


def new_value_generator(data_point, max_value, min_value):
    new_value = (data_point - min_value) / (max_value - min_value)
    return new_value


def data_normalizer(data):
    print 'Normalizing Data ...'
    for column_number in range(0, data.shape[1] - 1):
        max_value = get_max_col_value(data.iloc[0:, column_number])
        min_value = get_min_col_value(data.iloc[0:, column_number])
        data.iloc[:, column_number] = data.iloc[:, column_number].apply(new_value_generator, args=(max_value, min_value))
    return data


def k_fold_cross_validation(data, k_folds):
    print 'Creating Folds ...'
    number_of_examples = data.shape

    number_of_examples_per_partition = number_of_examples[0] / k_folds

    remainder = number_of_examples[0] % k_folds

    data['partition_no'] = np.nan

    row_lst = np.random.choice(number_of_examples[0], size=(1, number_of_examples[0]), replace=False).tolist()[0]

    if remainder == 0:
        for fold in range(0, k_folds):
            for i in range(0, number_of_examples_per_partition):
                data.set_value(row_lst[0], 'partition_no', fold)
                row_lst.pop(0)
    else:
        for fold in range(0, k_folds):
            if remainder > 0:
                compensation = 1
                remainder = remainder - 1
            else:
                compensation = 0

            for i in range(0, number_of_examples_per_partition + compensation):
                data.set_value(row_lst[0], 'partition_no', fold)
                row_lst.pop(0)
    return data


def get_folded_data(data, value):
    test_data = data.loc[data['partition_no'] == int(value)]
    test_data.drop('partition_no', axis=1, inplace=True)
    train_data = data.loc[data['partition_no'] != int(value)]
    train_data.drop('partition_no', axis=1, inplace=True)
    return train_data, test_data
