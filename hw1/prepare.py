from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np



def MinMaxScaling(train_data, new_data, columns):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_data[columns])
    new_data[columns] = scaler.transform(new_data[columns])
    return new_data


def StandardScaling(train_data, new_data, columns):
    scaler = StandardScaler()
    scaler.fit(train_data[columns])
    new_data[columns] = scaler.transform(new_data[columns])
    return new_data


def prepare_data(training_data, new_data):
    new_data_copy = new_data.copy()

    # Fill income's null values with Median.
    income_median = training_data['household_income'].median()
    new_data_copy['household_income'] = new_data_copy['household_income'].fillna(income_median)

    # Group blood types in A and B.
    new_data_copy["blood_type"] = new_data_copy["blood_type"].isin(["O+", "B+"])
    new_data_copy = new_data_copy.rename(columns={"blood_type": "SpecialProperty"})

    # Normalize features that need it.

    columns_to_minmax = ['PCR_03', 'PCR_04', 'PCR_09', 'PCR_10']
    new_data_copy = MinMaxScaling(training_data, new_data_copy, columns_to_minmax)

    columns_to_standard = ['PCR_01', 'PCR_02', 'PCR_05', 'PCR_06', 'PCR_07', 'PCR_08']
    new_data_copy = StandardScaling(training_data, new_data_copy, columns_to_standard)

    return new_data_copy
