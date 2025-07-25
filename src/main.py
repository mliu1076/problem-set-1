'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
from part1_etl import etl_data
from part2_preprocessing import preprocess_data
from part3_logistic_regression import logistic_regression
from part4_decision_tree import decision_tree_model
from part5_calibration_plot import calibration_plotting


# Call functions / instanciate objects from the .py files
def main():
    
    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    print("PART 1: ETL \n")
    etl_data()
    print("\n")

    # PART 2: Call functions/instanciate objects from preprocessing
    print("PART 2: Preprocessing \n")
    df_arrests = preprocess_data()
    print("\n")

    # PART 3: Call functions/instanciate objects from logistic_regression
    print("PART 3: Logistic Regression Model\n")
    df_arrests_lr = logistic_regression(df_arrests)
    print("\n")

    # PART 4: Call functions/instanciate objects from decision_tree
    print("PART 4: Decision Tree Model \n")
    df_arrests_dt = decision_tree_model(df_arrests)
    print("\n")

    # PART 5: Call functions/instanciate objects from calibration_plot
    print("PART 5: Calibration Plots + Extra Credit \n")
    c_plots = calibration_plotting(df_arrests_lr, df_arrests_dt)


if __name__ == "__main__":
    main()