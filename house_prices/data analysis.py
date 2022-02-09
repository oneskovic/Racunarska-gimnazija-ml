import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
data = pd.read_csv('data_normalized2.csv')
price = data.pop('SalePrice')

pca_components = 30

pca = PCA(n_components=pca_components)
res = pca.fit_transform(data)

for i in range(pca_components):
    plt.subplot(5,6,i+1)
    plt.scatter(res[:,i], price, s=3)
plt.show()

out = pd.DataFrame(res, columns=['PCA'+str(i) for i in range(pca_components)])
out['SalePrice'] = price
out.to_csv('data_pca.csv', index=False)


# for i,col in enumerate(data.columns[:20]):
#     plt.subplot(4,5,i+1)
#     plt.scatter(data[col],price, s=3)
#     plt.title(col)
# plt.show()
# plt.cla()

# for i,col in enumerate(data.columns[20:40]):
#     plt.subplot(4,5,i+1)
#     plt.scatter(data[col],price, s=3)
#     plt.title(col)
# plt.show()
# plt.cla()

# for i,col in enumerate(data.columns[40:60]):
#     plt.subplot(4,5,i+1)
#     plt.scatter(data[col],price, s=3)
#     plt.title(col)
# plt.show()
# plt.cla()

# for i,col in enumerate(data.columns[60:]):
#     plt.subplot(4,5,i+1)
#     plt.scatter(data[col],price, s=3)
#     plt.title(col)
# plt.show()
# plt.cla()


