import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, class_likelihood_ratios

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1])  # actual data
y_pred = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1])  # predicted values

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
'''
True Negatives (TN): 5
False Positives (FP): 1
False Negatives (FN): 2
True Positives (TP): 3

Row 1: Actual negatives - 5 correctly predicted as negative (TN), 1 incorrectly predicted as positive (FP)
Row 2: Actual positives - 2 incorrectly predicted as negative (FN), 3 correctly predicted as positive (TP)
'''

cr = classification_report(y_true, y_pred)
print("\nClassification Report:")
print(cr)

''' 
Precision formal:
    Actual positive / Actual Positive + False Positive 

Recall:
    Actual Positive / Actual Positive + False Negative
    
F1 Score:
    2 * Precision * Recall / Precision + Recall
'''

clr = class_likelihood_ratios(y_true, y_pred)
print(clr)





