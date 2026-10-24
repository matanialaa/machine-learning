
from kNN_test import kNN_test
from prepare import prepare_data
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # knn_obj = kNN_test()
    # knn_obj.test_Q21()
    dataset = pd.read_csv('virus_data.csv')
    train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=123)


    # Prepare training set according to itself
    train_df_prepared = prepare_data(train_df, train_df)

    # Prepare test set according to the raw training set
    test_df_prepared = prepare_data(train_df, test_df)

    outputPath_train = "train_df_prepared.csv"
    outputPath_test = "test_df_prepared.csv"

    train_df_prepared.to_csv(outputPath_train)
    test_df_prepared.to_csv(outputPath_test)
    # files.download(outputPath_train)
    # files.download(outputPath_test)
    #


if __name__ == '__main__':
    main()
