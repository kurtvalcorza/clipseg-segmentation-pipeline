reference = {0, 1, 2, 3}
prediction = {2, 3, 4, 5}
tp = len(reference & prediction)
fp = len(prediction - reference)
fn = len(reference - prediction)
print({'TP': tp, 'FP': fp, 'FN': fn, 'IoU': tp / (tp + fp + fn), 'Dice': 2 * tp / (2 * tp + fp + fn)})
