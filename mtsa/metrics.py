from sklearn import metrics

def calculate_aucroc(model, X_test, y_test):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.score_samples(X_test))
    auc = metrics.auc(fpr, tpr)
    return auc