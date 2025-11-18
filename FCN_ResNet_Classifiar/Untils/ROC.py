import numpy as np
from sklearn.metrics import roc_curve, auc,roc_auc_score
import matplotlib.pyplot as plt

y = np.array([1, 1, 1, 1, 1,
              2, 2, 2, 2, 2])
y_proba = np.array([0.1, 0.4, 0.4, 0.3, 0.5,
                    0.4, 0.6, 0.7, 0.8, 0.5])   # probability of prediction as positive

fpr, tpr, thresholds = roc_curve(y, y_proba, pos_label=2)
auc = auc(fpr, tpr)

roc_score=roc_auc_score(y,y_proba)
print(roc_auc_score)

plt.plot(fpr,tpr)
plt.show()
