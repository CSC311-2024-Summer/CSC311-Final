import regression
import decision_tree
import numpy as np














def display_performance_regression_and_decision_forest(df_of_size, race_size: int=6):
    # get models
    d = decision_tree.get_d_i(race_size)
    print('Created Decision Tree Model')
    r = regression.get_regression_model()
    print('Created Regression Model')
    X_datapoints, targets, static_vector = dataset
    print(static_vector)
    print(dataset[0])










if __name__ == '__main__':
    display_performance_regression_and_decision_forest(decision_tree.get_df(6))
