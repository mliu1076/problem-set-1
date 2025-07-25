'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

df_arrests = pd.read_csv('./data/df_arrests.csv')

def decision_tree_model(df_arrests):
    
    # creates a parameter grid for max_depth
    param_grid_dt = {
        'max_depth': [3, 5, 10]  # has three values for tree depths
    }

    # initializes the Decision Tree model
    dt_model = DTC(random_state=42)
    # initializes GridSearchCV using the Decision Tree model and parameter grid
    gs_cv_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=KFold_strat(5), scoring='accuracy')

    # separates features and target variable (assuming 'y' is the target)
    X = df_arrests[['num_fel_arrests_last_year', 'current_charge_felony']] 
    y = df_arrests['y'] 

    # splits the data into train and test sets (30% test size, stratified by 'y')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # fits the GridSearchCV model
    gs_cv_dt.fit(X_train, y_train)

    # gets the optimal max_depth
    optimal_max_depth = gs_cv_dt.best_params_['max_depth']
    print("Optimal value for max_depth: ", optimal_max_depth)

    # determines regularization
    if optimal_max_depth == 3:
        print("The optimal tree depth has the most regularization.")
    elif optimal_max_depth == 10:
        print("The optimal tree depth has the least regularization.")
    else:
        print("The optimal tree depth is in the middle (balanced regularization).")

    # predicts for the test set
    pred_dt_values = gs_cv_dt.predict(X_test)

    # assigns predictions to the corresponding rows in df_arrests using the indices of X_test
    df_arrests.loc[X_test.index, 'pred_dt'] = pred_dt_values

    # checks and fills in any missing predictions
    df_arrests['pred_dt'].fillna(0, inplace=True)
    # saves the updated dataframe as CSV in data folder
    df_arrests.to_csv('./data/df_arrests_with_dt_predictions.csv', index=False)

    # returns dataframe for use in part 5
    return df_arrests
