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

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

n_classes = len(np.unique(y))
n_components_lda = min(n_classes - 1, X.shape[1], 2)
lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
X_lda = lda.fit_transform(X, y)

# ── Elbow chart ────────────────────────────────────────────
pca_full = PCA().fit(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(cumvar) + 1), cumvar, marker='o', color='steelblue')
plt.axhline(0.95, linestyle='--', color='red',   label='95% variance')
plt.axhline(0.90, linestyle='--', color='orange', label='90% variance')
plt.xlabel("Số components"); plt.ylabel("Cumulative explained variance")
plt.title("PCA — Elbow Chart"); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_elbow.png', dpi=150, bbox_inches='tight')
plt.show()

n_95 = np.argmax(cumvar >= 0.95) + 1
n_90 = np.argmax(cumvar >= 0.90) + 1
print(f"\nPCA: cần {n_90} components để đạt 90%, {n_95} để đạt 95%")


# Convert PCA result to DataFrame
pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pca_columns)

# Add label
df_pca['NObeyesdad'] = y

# Save file
df_pca.to_csv('Dimensionality reduction/pca_data.csv', index=False)

print("Đã lưu PCA data!")



# Convert LDA result to DataFrame
lda_columns = [f'LD{i+1}' for i in range(X_lda.shape[1])]
df_lda = pd.DataFrame(X_lda, columns=lda_columns)

# Add label
df_lda['NObeyesdad'] = y

# Save file
df_lda.to_csv('Dimensionality reduction/lda_data.csv', index=False)

print("Đã lưu LDA data!")
