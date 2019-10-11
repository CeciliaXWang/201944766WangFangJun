from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import estimate_bandwidth

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


# #############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
# categories = None

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)


labels = dataset.target
true_k = np.unique(labels).shape[0]

t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       alternate_sign=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(dataset.data)


if opts.n_components:
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    explained_variance = svd.explained_variance_ratio_.sum()


# #############################################################################
# Do the actual clustering

km=KMeans(n_clusters=true_k, init='k-means++', n_init=1)
ap=AffinityPropagation()
bandwidth=estimate_bandwidth(X.toarray(), quantile=0.2, n_samples=500)
ms=MeanShift(bandwidth=bandwidth, bin_seeding=True)
sc=SpectralClustering(n_clusters=true_k, gamma=0.1,eigen_solver='arpack')
ac=AgglomerativeClustering(n_clusters=true_k,linkage='average')
whc=AgglomerativeClustering(n_clusters=true_k,linkage='ward')
db=DBSCAN()
gm=GaussianMixture(n_components=true_k)


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


bench(km,name="KM",data=X)
bench(ap,name="AP",data=X)
bench(ms,name="MS",data=X.toarray())
bench(sc,name="SC",data=X)
bench(ac,name="AC",data=X.toarray())
bench(whc,name="WHC",data=X.toarray())
bench(db,name="DB",data=X)
t0=time()
gm.fit(X.toarray())
print('%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f' %('GM',time()-t0,metrics.homogeneity_score(labels,gm.predict(X.toarray())),metrics.completeness_score(labels, gm.predict(X.toarray())), metrics.normalized_mutual_info_score(labels,gm.predict(X.toarray()))))

