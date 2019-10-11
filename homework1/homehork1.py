import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
#评价指标
from sklearn import metrics

#各种聚类方法
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture



from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from sklearn.preprocessing import scale
from time import time

digits=load_digits()
#获得原始数据
origin_data=digits.data
#获得原始数据的标签，即属于哪一类
labels=digits.target

#对原始数据进行标准化
data=scale(origin_data)
#查看label中一共有多少类
n_classes=len(np.unique(labels))


km=KMeans(init='random',n_clusters=10)
ap=AffinityPropagation()
ms=MeanShift()
sc=SpectralClustering(n_clusters=10, gamma=0.1)
ac=AgglomerativeClustering(n_clusters=10,linkage='average')
whc=AgglomerativeClustering(n_clusters=10,linkage='ward')
db=DBSCAN()
gm=GaussianMixture(n_components=10)


print(82 * '_')
print('name\t\ttime\t\th_score\t\tc_score\t\tnmi')

def bench(estimator, name, data):
    t0=time()
    estimator.fit(data)
    print('%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f'
         % (name,
            time()-t0,
            metrics.homogeneity_score(labels,estimator.labels_),
            metrics.completeness_score(labels, estimator.labels_),
            metrics.normalized_mutual_info_score(labels,estimator.labels_)
           ))


bench(km,name="KM",data=data)
bench(ap,name="AP",data=data)
bench(ms,name="MS",data=data)
bench(sc,name="SC",data=data)
bench(ac,name="AC",data=data)
bench(ac,name="WHC",data=data)
bench(db,name="DB",data=data)
t0=time()
gm.fit(data)
print('%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f' %('GM',time()-t0,metrics.homogeneity_score(labels,gm.predict(data)),metrics.completeness_score(labels, gm.predict(data)), metrics.normalized_mutual_info_score(labels,gm.predict(data))))
