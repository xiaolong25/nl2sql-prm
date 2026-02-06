# evaluation/metrics.py

def find_first_error_from_labels(labels):
    """
    labels: list[int] or 1D tensor, values in {0,1}
    return: int (0-based) or None
    """
    for i, v in enumerate(labels):
        if int(v) == 0:
            return i
    return None


def find_first_error_from_scores(scores, threshold=0.5):
    """
    scores: list[float] or 1D tensor
    return: int (0-based) or None
    """
    for i, s in enumerate(scores):
        if float(s) < threshold:
            return i
    return None
