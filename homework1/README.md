# Homework1： Clustering with sklearn  
## Data processing  
1、homework1中数据处理方式是先获得原始数据以及原始数据的标签（即属于哪一类），后对原始数据进行标准化  
```
digits=load_digits()
#获得原始数据
origin_data=digits.data
#获得原始数据的标签，即属于哪一类
labels=digits.target

#对原始数据进行标准化
data=scale(origin_data)
#查看label中一共有多少类
n_classes=len(np.unique(labels))

```
2、homework2中数据处理方式是使用sklearn中的TfidfVectorizer从文本列表中提取特征，得到向量表示矩阵(dim: [samples_num,features_num])  
```
vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(dataset.data)
```

## Clustering Algorithms
<div align=center>
  <img width="600" height="300" src="https://raw.githubusercontent.com/CeciliaXWang/201944766WangFangJun/master/homework1/picture/Clustering%20Algorithms.png"/>
</div>
在sklearn中使用上述聚类算法对数据集进行聚类
使用Homogeneity（hg_score）,Completeness(c_score),Normalized Mutual Information (NMI) 作为评价指标，对聚类效果进行评价
结果如下图
（1）homework1：digits手写字体数据集

（2）20newsgroups数据集
