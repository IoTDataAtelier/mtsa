from sklearn import metrics

def calculate_aucroc(model, X_test, y_test):
    y_hat=model.score_samples(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_hat)
    auc = metrics.auc(fpr, tpr)
    return auc