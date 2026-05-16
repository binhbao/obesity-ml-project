import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("Data/train_scale.csv")

X = data.drop(columns=['NObeyesdad']).values
y = data['NObeyesdad'].astype(int).values

n_classes = len(np.unique(y))
labels = [f"Class {i}" for i in range(n_classes)]
colors = plt.cm.tab10(np.linspace(0, 0.7, n_classes))

n_classes = len(np.unique(y))
n_components_lda = min(n_classes - 1, X.shape[1], 2)


lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
X_lda = lda.fit_transform(X, y)

print("\n=== LDA ===")
print(f"Explained variance ratio : {lda.explained_variance_ratio_}")
print(f"Tổng phương sai giữ lại  : {lda.explained_variance_ratio_.sum():.3f}")
