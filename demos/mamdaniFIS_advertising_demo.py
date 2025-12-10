import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

import sys
import os

# Add the library path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ex_fuzzy_reg.fuzzy_sets import FUZZY_SETS
from ex_fuzzy_reg.rules_reg_utils import generate_partitions
from ex_fuzzy_reg.regressors import MamdamiFIS

np.set_printoptions(suppress=True)


def mamdaniFIS_advertising_demo() -> None:
    # TV,Radio,Newspaper,Sales
    data = np.loadtxt('demos/datasets/advertising.csv', delimiter=',', skiprows=1)
    label_names = ["TV", "Radio", "Newspaper", "Sales"]
    fuzzy_variables = generate_partitions(data, n_labels=3, fv_label_names=label_names)
    print("Variables generated:")
    for i in range(len(fuzzy_variables)):
        print(fuzzy_variables[i].name)
        for j in range(len(fuzzy_variables[i].linguistic_variables)):
            print(fuzzy_variables[i].linguistic_variables[j])
        print()
    
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

    mamdaniFIS = MamdamiFIS(FUZZY_SETS.t1, linguistic_variables=fuzzy_variables)
    mamdaniFIS.fit(X_train, y_train)
    y_pred = mamdaniFIS.predict(X_test)

    print("First 10 samples:")
    print(np.hstack((X_test[:10], y_test[:10])))
    print(f"\nPredicted sales:\n {y_pred[:10]}")
    
    MSE = mean_squared_error(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)
    RMSE = root_mean_squared_error(y_test, y_pred)
    
    print(f"\nMean Squared Error: {MSE}")
    print(f"Mean Absolute Error: {MAE}")
    print(f"R² Score: {R2}")
    print(f"Root Mean Squared Error: {RMSE}")


if __name__ == '__main__':
    mamdaniFIS_advertising_demo()