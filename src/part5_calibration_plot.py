'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from part3_logistic_regression import logistic_regression
from part4_decision_tree import decision_tree_model
# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

def calibration_plotting(lr_arrests, dt_arrests):

    # gets the dataframe with predictions from the logistic regression and decision tree models
    df_arrests_lr = logistic_regression(lr_arrests)
    df_arrests_dt = decision_tree_model(dt_arrests)  # Assuming decision_tree_model is defined similarly

    # extracts columns from logistic regression model
    lr_pred_prob = df_arrests_lr['pred_lr']
    lr_y_true = df_arrests_lr['y']  

    # extracts the columns from decision tree model
    dt_pred_prob = df_arrests_dt['pred_dt']
    dt_y_true = df_arrests_dt['y'] 

    # creates calibration plot for logistic regression model
    print("Calibration curve for Logistic Regression:")
    calibration_plot(lr_y_true, lr_pred_prob, n_bins=5)

    # creates calibration plot for decision tree model
    print("Calibration curve for Decision Tree:")
    calibration_plot(dt_y_true, dt_pred_prob, n_bins=5)

    # calculates PPV for the top 50 arrestees ranked by predicted risk on the logistic regression model
    top_50_lr = df_arrests_lr.nlargest(50, 'pred_lr')
    ppv_lr = (top_50_lr['pred_lr'] == 1).sum() / 50
    print("PPV for Logistic Regression (Top 50 by predicted risk): ", ppv_lr)

    # calculates PPV for the top 50 arrestees ranked by predicted risk on the decision tree model
    top_50_dt = df_arrests_dt.nlargest(50, 'pred_dt')
    ppv_dt = (top_50_dt['pred_dt'] == 1).sum() / 50
    print("PPV for Decision Tree (Top 50 by predicted risk): ", ppv_dt)

    # calculates AUC for Logistic Regression
    auc_lr = roc_auc_score(lr_y_true, lr_pred_prob)
    print("AUC for Logistic Regression: ",auc_lr)

    # calculates AUC for Decision Tree
    auc_dt = roc_auc_score(dt_y_true, dt_pred_prob)
    print("AUC for Decision Tree: ", auc_dt)

    # compares AUCs
    if auc_lr > auc_dt:
        print("Logistic Regression is more accurate based on AUC.")
    elif auc_lr < auc_dt:
        print("Decision Tree is more accurate based on AUC.")
    else:
        print("Both models have the same accuracy based on AUC.")

