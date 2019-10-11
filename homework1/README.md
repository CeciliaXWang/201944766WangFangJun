Homework1： Clustering with sklearn  
Data processing  
1、homework1中数据处理方式是先获得原始数据以及原始数据的标签（即属于哪一类），后对原始数据进行标准化  
digits=load_digits()  
#获得原始数据  
origin_data=digits.data  
#获得原始数据的标签，即属于哪一类  
labels=digits.target  
#对原始数据进行标准化  
data=scale(origin_data)  
#查看label中一共有多少类  
n_classes=len(np.unique(labels))  
2、homework2中数据处理方式是使用sklearn中的TfidfVectorizer从文本列表中提取特征，得到向量表示矩阵(dim: [samples_num,features_num])  


Clustering Algorithms
