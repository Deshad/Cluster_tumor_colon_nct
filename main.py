import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, silhouette_score
import matplotlib.pyplot as plt

def init(vgg16_path):
    vgg16_content = h5py.File(vgg16_path, mode='r')

    # print(vgg16_content)
    vgg16_pca_feature  = vgg16_content['pca_feature'][...]
    # print(vgg16_pca_feature)
    vgg16_umap_feature  = vgg16_content['umap_feature'][...]
    # print("Shape of vgg16_pca_feature:", vgg16_pca_feature.shape)
    filename  = np.squeeze(vgg16_content['file_name'])
    filename = np.array([str(x) for x in filename])
    labels = np.array([x.split('/')[2] for x in filename])
    print(labels)
    print(len(vgg16_umap_feature))
    random.seed(0)
    # selected_index = random.sample(list(np.arange(len(vgg16_pca_feature))), 200)
    selected_index = random.sample(list(np.arange(len(vgg16_umap_feature))), 200)

    test_data = vgg16_pca_feature[selected_index]
    test_label = labels[selected_index]

    return test_data, test_label, labels


def plot3D(labels, test_data, test_label):
        traces = []
        for name in np.unique(labels):
            trace = go.Scatter3d(
                x=test_data[test_label==name,0],
                y=test_data[test_label==name,1],
                z=test_data[test_label==name,2],
                mode='markers',
                name=name,
                marker=go.scatter3d.Marker(
                    size=4,
                    opacity=0.8
                )

            )
            traces.append(trace)


        data = go.Data(traces)
        layout = go.Layout(
                    showlegend=True,
            scene=go.Scene(
                        xaxis=go.layout.scene.XAxis(title='PC1'),
                        yaxis=go.layout.scene.YAxis(title='PC2'),
                        zaxis=go.layout.scene.ZAxis(title='PC3')
                        )
        )
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            title="First 3 pricipal components of VGG16's PCA feature",
            legend_title="Legend Title",
        )
        fig.show()

def model_training(test_data):
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans_assignment = kmeans.fit_predict(test_data)
    # adjacency_matrix = sparse.csr_matrix(MinMaxScaler().fit_transform(-pairwise_distances(test_data)))
    # louvain_assignment = louvain_model.fit_transform(adjacency_matrix)

    return kmeans_assignment

def eval_vis(kmeans_assignment, test_data, test_label):
    print('Number of clusters from KMeans: %d' % np.unique(kmeans_assignment).shape[0])
    kmeans_counts = np.unique(kmeans_assignment, return_counts=True)

    print('Kmeans assignment counts')
    print(pd.DataFrame({'Cluster Index': kmeans_counts[0], 'Number of members': kmeans_counts[1]}).set_index('Cluster Index'))
    
    # Calculate metrics
    kmeans_silhouette = silhouette_score(test_data, kmeans_assignment)
    kmeans_v_measure = v_measure_score(test_label, kmeans_assignment)
    metrics_df = pd.DataFrame({'Metrics': ['silhouette', 'V-measure'], 'Kmeans': [kmeans_silhouette, kmeans_v_measure]})
    print(metrics_df.set_index('Metrics'))
    
    # Plot metrics
    metrics_df.set_index('Metrics').plot(kind='bar', legend=False, figsize=(8, 5), color=['skyblue'])
    plt.title("KMeans Clustering Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def calculate_percent(sub_df, attrib):
    cnt = sub_df[attrib].count()
    output_sub_df = sub_df.groupby(attrib).count()
    return (output_sub_df/cnt)

def visual(kmeans_assignment, test_label):
    resulted_cluster_df = pd.DataFrame({'clusterID': kmeans_assignment, 'type': test_label})
    label_proportion_df = resulted_cluster_df.groupby(['clusterID']).apply(lambda x: calculate_percent(x,'type')).rename(columns={'clusterID':'type_occurrence_percentage'}).reset_index()
    pivoted_label_proportion_df = pd.pivot_table(label_proportion_df, index = 'clusterID', columns = 'type', values = 'type_occurrence_percentage')


    plt.figure(figsize=(10,5))
    # number_of_tile_df = resulted_cluster_df.groupby('clusterID')['type'].count().reset_index().rename(columns={'type':'number_of_tile'})
    df_idx = pivoted_label_proportion_df.index
    (pivoted_label_proportion_df*100).loc[df_idx].plot.bar(stacked=True)

    plt.ylabel('Percentage of tissue type')
    plt.legend(loc='upper right')
    plt.title('Cluster configuration by Kmeans')
    plt.show()

def main():
    vgg16_path = "data/vgg16_dim_reduced_feature.h5"
    test_data, test_label, labels = init(vgg16_path=vgg16_path)
    # plot3D(labels, test_data, test_label)
    kmeans_assignment = model_training(test_data=test_data)
    eval_vis(kmeans_assignment, test_data, test_label)
    visual(kmeans_assignment=kmeans_assignment, test_label=test_label)


if __name__ == "__main__":
    main()