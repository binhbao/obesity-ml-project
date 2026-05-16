import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
from scipy.spatial.distance import cdist

# --- Load ---
df = pd.read_csv(r"Dimensionality reduction/pca_data.csv")

X = df.drop(columns=['NObeyesdad']).values
y = df['NObeyesdad']

# ===== 1. CHỌN K TỐI ƯU =====
Ks = range(2, 10)
sil_scores = []

for k in Ks:
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(X)
    sil_scores.append(silhouette_score(X, labels))

plt.figure()
plt.plot(Ks, sil_scores, marker='o')
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Chọn K tối ưu")
plt.show()

# 👉 chọn K tốt nhất
K = Ks[np.argmax(sil_scores)]
print(f"\nK tối ưu theo Silhouette: {K}")

# ===== 2. KMEANS =====
kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
labels_km = kmeans.fit_predict(X)

# ===== 3. ĐÁNH GIÁ NỘI TẠI =====
silhouette = silhouette_score(X, labels_km)
davies_bouldin = davies_bouldin_score(X, labels_km)
calinski_harabasz = calinski_harabasz_score(X, labels_km)

print("\n" + "=" * 50)
print("       ĐÁNH GIÁ NỘI TẠI (Unsupervised)")
print("=" * 50)
print(f"Silhouette Score     : {silhouette:.3f}  (↑ tốt)")
print(f"Davies-Bouldin Index : {davies_bouldin:.3f}  (↓ tốt)")
print(f"Calinski-Harabasz    : {calinski_harabasz:.1f} (↑ tốt)")

# ===== 4. ĐÁNH GIÁ NGOÀI (có nhãn thật) =====
ari = adjusted_rand_score(y, labels_km)

print("\n" + "=" * 50)
print("       ĐÁNH GIÁ NGOÀI (So với nhãn thật)")
print("=" * 50)
print(f"Adjusted Rand Index (ARI): {ari:.3f}")

# ===== 5. QUAN HỆ GIỮA CÁC CỤM =====
centers = kmeans.cluster_centers_
dist_matrix = cdist(centers, centers)
avg_dist = np.mean(dist_matrix[np.triu_indices(K, 1)])

print("\n" + "=" * 50)
print("          QUAN HỆ GIỮA CÁC CỤM")
print("=" * 50)
print("Khoảng cách giữa các cụm:\n", np.round(dist_matrix, 2))
print(f"Khoảng cách trung bình: {avg_dist:.2f}")

# ===== 6. PHÂN BỐ NHÃN TRONG CỤM =====
print("\n" + "=" * 50)
print("     BẢNG CHÉO: CLUSTER vs NHÃN THẬT")
print("=" * 50)
print(pd.crosstab(labels_km, y))

# ===== 7. VISUAL =====
plt.figure(figsize=(6, 5))
sc = plt.scatter(X[:, 0], X[:, 1], c=labels_km, cmap='tab10')

plt.scatter(
    centers[:, 0], centers[:, 1],
    marker='x', s=120, c='red', label='Centroids'
)

plt.title(f"K-means | K={K} | Sil={silhouette:.2f} | DB={davies_bouldin:.2f}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(sc, label="Cluster")
plt.legend()
plt.tight_layout()
plt.show()

# ===== 8. TỔNG KẾT =====
print("\n" + "=" * 50)
print("               TỔNG KẾT")
print("=" * 50)
print(f"K tối ưu              : {K}")
print(f"Silhouette Score      : {silhouette:.3f}")
print(f"Davies-Bouldin Index  : {davies_bouldin:.3f}")
print(f"Calinski-Harabasz     : {calinski_harabasz:.1f}")
print(f"ARI                   : {ari:.3f}")
print(f"Khoảng cách cụm TB    : {avg_dist:.2f}")