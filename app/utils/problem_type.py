from sklearn.utils.multiclass import type_of_target


def detect_problem_type(y):
    """
    Detects whether the problem is classification, regression, clustering,
    or anomaly detection.
    """
    target_type = type_of_target(y)

    if target_type in ["binary", "multiclass", "multilabel-indicator"]:
        return "classification"
    elif target_type == "continuous":
        return "regression"
    elif target_type == "continuous-multioutput":
        return "multi-output regression"
    elif target_type == "multiclass-multioutput":
        return "multi-label classification"
    else:
        return "unsupervised"
