import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

def propertyFunc_var(adj_matrix):
    return adj_matrix.var()

def propertyFunc_mean(adj_matrix):
    return adj_matrix.mean()

def propertyFunc_std(adj_matrix):
    return adj_matrix.std()

def propertyFunc_UnidirectionalIndex(adj_matrix):
    unidirectionalIndex = 0
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[0])):
            unidirectionalIndex = unidirectionalIndex +\
                                  abs(adj_matrix[i][j] - adj_matrix[j][i])
    return unidirectionalIndex

def getPropertyArrayWithPropertyFunc(data_input, property_func):
    result = []
    for i in range(len(data_input)):
        result.append(property_func(data_input[i]))
    # -- standardlize
    return np.array(result)

def getDistanceMatrixWithPropertyArray(data_x, property_array, isSigmoid=False):
    sampleNum = data_x.shape[0]
    disMatrix = np.zeros([sampleNum, sampleNum])
    for i in range(0, sampleNum):
        for j in range(0, sampleNum):
            if isSigmoid:
                hour_i = i % 24
                hour_j = j % 24
                hour_dis = abs(hour_i-hour_j)
                if hour_dis == 23:
                    hour_dis = 1
                c = sigmoid(hour_dis/24)
            else:
                c = 1
            disMatrix[i][j] = c * abs(property_array[i] - property_array[j])
    disMatrix = (disMatrix - disMatrix.min()) / (disMatrix.max() - disMatrix.min())
    return disMatrix

def getDistanceMatrixWithPropertyFunc(data_x, property_func, isSigmoid=False):
    property_array = getPropertyArrayWithPropertyFunc(data_x, property_func)
    disMatrix = getDistanceMatrixWithPropertyArray(data_x, property_array, isSigmoid=isSigmoid)
    return disMatrix

def get_SSEncode2D(one_data, mean_data):
    result = []
    for i in range(len(one_data)):
        for j in range(len(one_data[0])):
            if one_data[i][j] > mean_data[i][j]:
                result.append(1)
            else:
                result.append(0)
    return np.array(result)

def getDistanceMatrixWith_SSIndex(input_data, isSigmoid=True):
    sampleNum = len(input_data)
    input_data_mean = input_data.mean(axis=0)
    property_array = []
    for i in range(len(input_data)):
        property_array.append(get_SSEncode2D(input_data[i], input_data_mean))
    property_array = np.array(property_array)
    disMatrix = np.zeros([sampleNum, sampleNum])
    for i in range(0, sampleNum):
        for j in range(0, sampleNum):
            if isSigmoid:
                hour_i = i % 24
                hour_j = j % 24
                sub_hour = abs(hour_i-hour_j)
                if sub_hour == 23:
                    sub_hour = 1
                c = sigmoid(sub_hour/24)
            else:
                c = 1
            sub_encode = abs(property_array[i] - property_array[j])
            disMatrix[i][j] = c * sub_encode.sum()
    disMatrix = (disMatrix - disMatrix.min()) / (disMatrix.max() - disMatrix.min())
    label_pred = getClusterLabelWithDisMatrix(disMatrix, display_dis_matrix=False)
    return disMatrix

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Mobility_Graph_Distance(m_graphs):
    """
    :param m_graphs: (N, M, M).  N graphs, each graph has M nodes
    :return: (N, N). Distance matrix between every two graphs
    """
    # Mean
    isSigmoid = True
    mean_dis_matrix = getDistanceMatrixWithPropertyFunc(
        m_graphs, propertyFunc_mean, isSigmoid=isSigmoid)
    # Uniflow
    unidirIndex_dis_matrix = getDistanceMatrixWithPropertyFunc(
        m_graphs, propertyFunc_UnidirectionalIndex, isSigmoid=isSigmoid
    )
    # Var
    var_dis_matrix = getDistanceMatrixWithPropertyFunc(
        m_graphs, propertyFunc_var, isSigmoid=isSigmoid
    )
    # SS distance
    ss_dis_matrix = getDistanceMatrixWith_SSIndex(m_graphs, isSigmoid=isSigmoid)
    c_mean_dis = 1
    c_unidirIndex_dis = 1
    c_std_dis = 1
    c_ss_dis = 1
    disMatrix = (c_mean_dis * mean_dis_matrix) \
                + (c_unidirIndex_dis * unidirIndex_dis_matrix) \
                + (c_std_dis * var_dis_matrix) \
                + (c_ss_dis * ss_dis_matrix)
    return disMatrix

def getClusterLabelWithDisMatrix(dis_matrix, display_dis_matrix=False):
    n_clusters = 7
    # # linkage: single, average, complete
    linkage = "complete"
    # ---
    # t1 = time.time()
    if display_dis_matrix:
        sns.heatmap(dis_matrix)
        plt.show()
    # ---
    estimator = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage, affinity="precomputed", )
    estimator.fit(dis_matrix)
    label_pred = estimator.labels_
    # print("The time consuming of clustering (known disMatrix)：", time.time() - t1)
    return label_pred

def getPatternWithMGD(m_graphs):
    """
    :param m_graphs: (N, M, M).  N graphs, each graph has M nodes
    :return mob_patterns:
    :return cluster_label: 
    """
    n_clusters = 7
    linkage = "complete"
    disMatrix = Mobility_Graph_Distance(m_graphs)
    # -- Agglomerative Cluster
    estimator = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity="precomputed", )
    estimator.fit(disMatrix)
    label_pred = estimator.labels_
    cluster_label = label_pred.reshape((31, 24))
    # -- Generate Mobility Pattern
    patterns = []
    for i in range(n_clusters):
        this_cluster_index_s = np.argwhere(label_pred == i).flatten()
        this_cluster_graph_s = m_graphs[this_cluster_index_s]
        patterns.append(this_cluster_graph_s.sum(axis=0))
    mob_patterns = np.array(patterns)
    return mob_patterns, cluster_label

if __name__ == '__main__':
    multi_graph = np.load("./data/mob_multi_graph.npy")
    mob_patterns, cluster_label = getPatternWithMGD(multi_graph)
    np.save("./data/mob_pattern.npy", mob_patterns)

