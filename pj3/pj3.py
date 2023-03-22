import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from yellowbrick.cluster import KElbowVisualizer
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
sns.set()

def plot_cluster_vs_metric(df,cluster_col_name, metric_col_names,dataset_name, algo_name):
    for metric in metric_col_names:
        plt.plot(df[cluster_col_name], df[metric], label = metric)
    plt.legend()
    plt.savefig("result/{}-{} metrics.png".format(dataset_name, algo_name))
    plt.clf()

def find_ICA_best_param(data_x):
    dims = list(np.arange(2, (data_x.shape[1] - 1)))
    dims.append(data_x.shape[1])
    ica = FastICA(random_state=5)
    kurt = []
    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(data_x)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.abs().mean())
    return dims, kurt
def plot_ICA(dims,kurt,dataset_name):
    plt.figure()
    plt.title("ICA Kurtosis: {}".format(dataset_name))
    plt.xlabel("Independent Component Numbers")
    plt.ylabel("Avg Kurtosis")
    plt.plot(dims, kurt, 'b-')
    plt.savefig("result/ICA Kurtosis-{}.png".format(dataset_name))
    plt.clf()
def randomized_projection(X, k):
    """Reduce the dimensionality of data X to k dimensions using randomized projection."""
    n, d = X.shape
    R = np.random.normal(size=(d, k))   # Generate a random projection matrix
    Y = np.dot(X, R)                    # Project the data onto the subspace
    X_hat = np.dot(Y, R.T)              # Reconstruct the data in the original space
    return Y, X_hat

data_wine=pd.read_csv("winequality-white.csv")
data_iris = pd.read_csv('iris.csv')
data_default = pd.read_csv('Italy_default_prior_2004.csv')
dataset_name='default' # 'iris, wine
process = 'NN+clustering' # 'clustering','PCA', 'ICA', 'Randomized projection', 'Forward stepwise', 'NN+PCA', 'NN+clustering'

if dataset_name=='wine':
    data=data_wine.sample(frac=1).reset_index(drop=True)#shuffle data
    data_x=data.loc[:,data.columns!="quality"]
    data_y=data.loc[:,data.columns=="quality"]
if dataset_name == 'iris':
    cleanup_type = {"type": {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}}
    data_iris = data_iris.replace(cleanup_type)
    data = data_iris
    data_x = data.loc[:, data.columns != "type"]
    data_y = data.loc[:, data.columns == "type"]
if dataset_name == 'default':
    data = data_default.sample(frac=1).reset_index(drop=True)  # shuffle data
    data_x = data.loc[:, data.columns != "DefaultIndex"]
    le = LabelEncoder()
    for column in data_x.columns:
        data_x[column] = le.fit_transform(data_x[column])
    data_y = data.loc[:, data.columns == "DefaultIndex"].astype('int')
x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.4,train_size=0.6,random_state=0).copy()
data_y_array = np.array(data_y.iloc[:,0].T)
if process == 'clustering':
    # K Means
    #sum of squared distance
    sum_squared_distances = []
    K = range(2,20)
    for k in K:
        km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km = km.fit(data_x)
        sum_squared_distances.append([km.inertia_, k, dataset_name])
    elbow_df = pd.DataFrame(sum_squared_distances, columns=["sum_squared_distance", "k", "Dataset"])
    sns.lineplot(x="k", y="sum_squared_distance", data=elbow_df, marker="o").set_title("{}: Sum of Squared Distance".format(dataset_name))
    plt.savefig('result/{}-kmeans sum sq distance.png'.format(dataset_name))
    plt.clf()
    #silhouette score
    model = KMeans(random_state=0)
    visualizer = KElbowVisualizer(model, k=(2,20), metric='silhouette', timings=False)
    visualizer.fit(data_x)
    plt.savefig("result/{}-kmeans Silhouette Score.png".format(dataset_name))
    plt.clf()


    # Expecation Maximization
    silhouette_score = []
    K = range(2,20)
    for k in K:
        EM = GaussianMixture(n_components= k, random_state=0).fit(data_x).predict(data_x)
        silhouette_score.append([metrics.silhouette_score(data_x, EM, metric='euclidean', sample_size=50), k, dataset_name])
    silhouette_df = pd.DataFrame(silhouette_score, columns=["silhouette_score", "k", "Dataset"])
    sns.lineplot(x="k", y="silhouette_score", hue="Dataset", data=silhouette_df, marker="o")
    plt.savefig('result/{}-EM Silhouette Score.png'.format(dataset_name))
    plt.clf()

    #Compare Kmeans vs EM
    clusters = [x for x in range(2, 20)]
    metricOutputs_kmeans = []
    metricOutputs_EM = []

    for k in clusters:
        KMeans_model = KMeans(init='k-means++', n_clusters=k, n_init=10)
        KMeans_model.fit(data_x)
        Y_p = KMeans_model.predict(data_x)
        metricOutputs_kmeans.append([
                                    dataset_name,
                                    "Kmeans",
                                    k,
                                    metrics.homogeneity_score(data_y_array, Y_p),
                                    metrics.completeness_score(data_y_array, Y_p),
                                    metrics.silhouette_score(data_x, Y_p, metric='euclidean', sample_size=50)
                                    ]  )

        EM_model = GaussianMixture(n_components=k,random_state=0)
        EM_model.fit(data_x)
        Y_p = EM_model.predict(data_x)
        metricOutputs_EM.append([
            dataset_name,
            'Expectation Maximization',
            k,
            metrics.homogeneity_score(data_y_array, Y_p),
            metrics.completeness_score(data_y_array, Y_p),
            metrics.silhouette_score(data_x, Y_p, metric='euclidean', sample_size=50)
        ])
    kmeans_metrics_df = pd.DataFrame(metricOutputs_kmeans, columns=[
        "Dataset",
        "Algo",
        "Number of Clusters",
        "homogeneity_score",
        "completeness_score",
        "silhouette_score"
    ])

    EM_metrics_df = pd.DataFrame(metricOutputs_EM, columns=[
        "Dataset",
        "Algo",
        "Number of Clusters",
        "homogeneity_score",
        "completeness_score",
        "silhouette_score"
    ])

    plot_cluster_vs_metric(kmeans_metrics_df,"Number of Clusters",["homogeneity_score","completeness_score","silhouette_score"],dataset_name,'kmeans')
    plot_cluster_vs_metric(EM_metrics_df,"Number of Clusters",["homogeneity_score","completeness_score","silhouette_score"],dataset_name,'EM')

#######Dimension Reduction
if process == 'PCA':
    #PCA
    pca = PCA().fit(data_x)
    plt.plot(np.cumsum(pca.explained_variance_ratio_)) #2 components will explain 98% for wine and 98% for iris
    plt.xlabel('components number')
    plt.ylabel('cumulative explained variance')
    plt.savefig('result/PCA cum explained Variance-{}.png'.format(dataset_name))
    plt.clf()


    pca = PCA(2)  # project to 2 dimensions
    PCA_projected = pca.fit_transform(data_x)
    plt.scatter(PCA_projected[:, 0], PCA_projected[:, 1],
                c=data_y_array, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Spectral', 5))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.savefig('result/PC1vs2-{}.png'.format(dataset_name))
    plt.clf()
    print(dataset_name + " PCA explained_variance is: " + str(pca.explained_variance_ratio_))
    print(dataset_name + " PCA cumulative explained_variance is: " + str(np.cumsum(pca.explained_variance_ratio_)))

    #Perform Clustering after PCA
    sum_squared_distances = []
    K = range(2, 20)
    for k in K:
        km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km = km.fit(PCA_projected)
        sum_squared_distances.append([km.inertia_, k, dataset_name])
    elbow_df = pd.DataFrame(sum_squared_distances, columns=["sum_squared_distance", "k", "Dataset"])
    sns.lineplot(x="k", y="sum_squared_distance", data=elbow_df, marker="o").set_title(
        "{}: Sum of Squared Distance".format(dataset_name))
    plt.savefig('result/{}-kmeans+PCA sum sq distance.png'.format(dataset_name))
    plt.clf()
if process == 'ICA':
    #ICA
    #for wine dataset, 2 component can have significant higher than 3 Kurtosis, for IRIS, cannot find a good enough combination
    dims,curt = find_ICA_best_param(data_x)
    plot_ICA(dims,curt,dataset_name)
    #using 2 component to test going forward
    ica = FastICA(n_components = data_x.shape[1])
    ica.fit(data_x)
    ICA_projected = ica.transform(data_x)
    ax = sns.barplot(
        x=np.arange(1,len(kurtosis(ICA_projected))+1,1),
        y=kurtosis(ICA_projected)
    )
    ax.set(xlabel='Features', ylabel='Kurtosis')
    ax.set_title('Kurtosis distribution for {}'.format(dataset_name))
    plt.savefig('result/Kurtosis distribution for {}.png'.format(dataset_name))
    plt.clf()

    # Perform Clustering after ICA
    sum_squared_distances = []
    K = range(2, 20)
    for k in K:
        km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km = km.fit(ICA_projected)
        sum_squared_distances.append([km.inertia_, k, dataset_name])
    elbow_df = pd.DataFrame(sum_squared_distances, columns=["sum_squared_distance", "k", "Dataset"])
    sns.lineplot(x="k", y="sum_squared_distance", data=elbow_df, marker="o").set_title(
        "{}: Sum of Squared Distance".format(dataset_name))
    plt.savefig('result/{}-kmeans+ICA sum sq distance.png'.format(dataset_name))
    plt.clf()

if process == 'Randomized projection':
    #Randomized projection
    random_projection_result=[]
    for i in range(data_x.shape[1]):
        Y, x_hat = randomized_projection(data_x, i)       # Reduce the dimensionality to 5 dimensions
        recon_error = np.linalg.norm(data_x - x_hat, ord='fro') / np.linalg.norm(data_x, ord='fro')
        corr_coef = np.corrcoef(data_x.T, Y.T)[0, 1]
        random_projection_result.append([
            dataset_name,
            i,
            recon_error,
            corr_coef
        ])
    random_projection_result = pd.DataFrame(random_projection_result,columns=['dataset_name','reduced dimension',
                                                                              'Reconstruction error','Correlation coefficient']).set_index('reduced dimension')
    random_projection_result['Reconstruction error'].plot()
    plt.savefig('result/random projection reconstruction error {}.png'.format(dataset_name))
    plt.clf()
    random_projection_result['Correlation coefficient'].plot()
    plt.savefig('result/random projection Correlation coefficient {}.png'.format(dataset_name))
    plt.clf()

    #Apply clustering after randomized projection
    Y, x_hat = randomized_projection(data_x, 2) # using 2 dimension
    sum_squared_distances = []
    K = range(2, 20)
    for k in K:
        km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km = km.fit(x_hat)
        sum_squared_distances.append([km.inertia_, k, dataset_name])
    elbow_df = pd.DataFrame(sum_squared_distances, columns=["sum_squared_distance", "k", "Dataset"])
    sns.lineplot(x="k", y="sum_squared_distance", data=elbow_df, marker="o").set_title(
        "{}: Sum of Squared Distance".format(dataset_name))
    plt.savefig('result/{}-kmeans+RP sum sq distance.png'.format(dataset_name))
    plt.clf()

if process == 'Forward stepwise':
    #Forward stepwise variable selection
    clf_nn = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000)
    selected_features = []
    accuracy_scores = []
    remaining_features = list(x_train.columns)
    max_nn_accuracy = {k:-np.inf for k in range(1, len(remaining_features)+1)}
    max_accuracy_feature = {}
    i=0
    while len(remaining_features) > 0 and i<len(x_train.columns):
        i += 1
        for feature in remaining_features:
            clf_nn.fit(x_train[selected_features + [feature]], y_train)
            y_pred = clf_nn.predict(x_test[selected_features + [feature]])
            nn_accuracy = accuracy_score(y_test, y_pred)
            if nn_accuracy > max_nn_accuracy[len(selected_features)+1]:
                max_nn_accuracy[len(selected_features)+1] = nn_accuracy
                max_accuracy_feature[len(selected_features)+1] = feature
        remaining_features.remove(max_accuracy_feature[len(selected_features)+1])
        selected_features.append(max_accuracy_feature[len(selected_features)+1])
        print("Selected {} as {}th variable".format(selected_features[-1], str(i)))
    fig, ax = plt.subplots()
    ax.plot(max_accuracy_feature.values(), max_nn_accuracy.values(), label='Accuracy')
    ax.legend()
    ax.set_xlabel('Feature Name')
    ax.set_ylabel('Accuracy')
    plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='right')
    plt.savefig('result/{} forward Stepwise metrics.png'.format(dataset_name))
    plt.clf()

    #Apply clustering after forward selection
    data_x_selected = data_x[list(max_accuracy_feature.values())[0:2]]
    sum_squared_distances = []
    K = range(2, 20)
    for k in K:
        km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km = km.fit(data_x_selected)
        sum_squared_distances.append([km.inertia_, k, dataset_name])
    elbow_df = pd.DataFrame(sum_squared_distances, columns=["sum_squared_distance", "k", "Dataset"])
    sns.lineplot(x="k", y="sum_squared_distance", data=elbow_df, marker="o").set_title(
        "{}: Sum of Squared Distance".format(dataset_name))
    plt.savefig('result/{}-kmeans+Forward sum sq distance.png'.format(dataset_name))
    plt.clf()

if process == 'NN+PCA':
###############rerun neural network on PCA reduced dataset
    #Split dataset into train and test dataset
    pca = PCA(4)  # project to 2 dimensions
    PCA_projected = pca.fit_transform(data_x)
    x_train, x_test, y_train, y_test = train_test_split(PCA_projected, data_y, test_size=0.4, train_size=0.6,
                                                    random_state=0).copy()

    # ica = FastICA(n_components = data_x.shape[1])
    # ica.fit(data_x)
    # ICA_projected = ica.transform(data_x)
    # x_train, x_test, y_train, y_test = train_test_split(ICA_projected, data_y, test_size=0.4, train_size=0.6,
    #                                                 random_state=0).copy()
    clf_nn = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000)
    clf_nn.fit(x_train, y_train)
    y_predict = clf_nn.predict(x_test)
    nn_accuracy = accuracy_score(y_test, y_predict)
    print(dataset_name + ' Accuracy of NN after PCA is %.2f%%' % (nn_accuracy * 100))

if process == 'NN+clustering':
    k = len(data_y.iloc[:,0].unique()) #use number of unique value in y-label as k
    km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    km = km.fit(data_x)
    data_x['assigned_cluster'] = km.predict(data_x)
    x_train,x_test,y_train,y_test=train_test_split(data_x ,data_y,test_size=0.4,train_size=0.6,random_state=0).copy()
    clf_nn = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000)
    clf_nn.fit(x_train, y_train)
    y_predict = clf_nn.predict(x_test)
    nn_accuracy = accuracy_score(y_test, y_predict)
    print(dataset_name + ' Accuracy of NN after K-Mmeans is %.2f%%' % (nn_accuracy * 100))
print('finished')