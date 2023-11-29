# Import the necessary libraries used in the functions created
import matplotlib.pyplot as plt
# Import all metrics for classification task
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score


# Function to calculate metrics
def calculate_metrics(model, X_test, y_test):
    '''
    The calculate_metrics is a function that takes as input: model, X_test, y_test 
    and returns a tuple of metrics (accuracy, precision, recall, f1 score)
    '''
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate the AUC Value
    y_pred_prob = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, probabilities)
    return accuracy, precision, recall, f1, auc



# Function to plot ROC Curve for a given model
def plot_roc_curve(model, X_test, y_test, model_name):
     '''
    The plot_roc_curve is a function that takes as input: model, X_test, y_test, model_name
    and returns the ROC Curve plot with the AUC value
    '''
    # ROC Curve and AUC
    probabilities = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)
    
    # Plotting ROC Curve
    plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_name} ROC Curves")
    plt.legend(loc='lower right')
    plt.show()