import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist

# --- Load ---
df = pd.read_csv(r"Dimensionality reduction/pca_data.csv")

X = df.drop(columns=['NObeyesdad']).values
y = df['NObeyesdad']

# ===== 1. CHỌN EPS (k-distance plot) =====
k = 5  # = min_samples
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, _ = nbrs.kneighbors(X)

# Lấy khoảng cách tới hàng xóm thứ k
k_dist = np.sort(distances[:, -1])

plt.figure()
plt.plot(k_dist)
plt.title("K-distance graph (chọn eps)")
plt.xlabel("Points sorted")
plt.ylabel("Distance")
plt.show()


eps = 1.5   # <-- chỉnh theo plot

# ===== 2. DBSCAN =====
db = DBSCAN(eps=eps, min_samples=k)
labels_db = db.fit_predict(X)

# ===== 3. THÔNG TIN CỤM =====
n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = np.sum(labels_db == -1)

print("=" * 50)
print("             THÔNG TIN DBSCAN")
print("=" * 50)
print(f"Số cụm tìm được : {n_clusters}")
print(f"Số điểm nhiễu   : {n_noise}")

# ===== 4. ĐÁNH GIÁ =====
mask = labels_db != -1

if len(set(labels_db[mask])) > 1:
    silhouette = silhouette_score(X[mask], labels_db[mask])
else:
    silhouette = -1

ari = adjusted_rand_score(y, labels_db)

print("\n" + "=" * 50)
print("               ĐÁNH GIÁ")
print("=" * 50)
print(f"Silhouette Score (không tính nhiễu): {silhouette:.3f}")
print(f"Adjusted Rand Index (ARI): {ari:.3f}")

# ===== 5. NỘI CỤM =====
print("\n" + "=" * 50)
print("           KHOẢNG CÁCH NỘI CỤM")
print("=" * 50)

clusters = [k for k in set(labels_db) if k != -1]

for k in clusters:
    Xk = X[labels_db == k]
    if len(Xk) > 1:
        intra = np.mean(pdist(Xk))
        print(f"Cluster {k}: {round(intra, 2)}")

# ===== 6. PHÂN BỐ NHÃN =====
print("\n" + "=" * 50)
print("     BẢNG CHÉO: CLUSTER vs NHÃN")
print("=" * 50)
print(pd.crosstab(labels_db, y))

# ===== 7. VISUAL =====
# Nếu PCA của bạn đã 2D thì không cần PCA lại
if X.shape[1] > 2:
    from sklearn.decomposition import PCA
    X_vis = PCA(n_components=2).fit_transform(X)
else:
    X_vis = X

plt.figure(figsize=(6,5))

# Vẽ cluster
sc = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels_db, cmap='tab10')

# Highlight nhiễu
noise_mask = labels_db == -1
plt.scatter(
    X_vis[noise_mask, 0],
    X_vis[noise_mask, 1],
    c='black',
    s=10,
    label='Noise'
)

plt.title(f"DBSCAN | eps={eps} | clusters={n_clusters}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.colorbar(sc)
plt.tight_layout()
plt.show()

# ===== 8. NHẬN XÉT =====
print("\n" + "=" * 50)
print("               NHẬN XÉT")
print("=" * 50)
print("- DBSCAN không cần chọn K.")
print("- eps quyết định rất mạnh kết quả.")
print("- Có khả năng phát hiện nhiễu (-1).")
print("- Nếu quá nhiều nhiễu → eps nhỏ hoặc dữ liệu phân tán.")
print("- Nếu chỉ có 1 cụm → eps quá lớn.")