# main program
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

class AIME:
    def __init__(self):
        self.A_dagger = None
        self.scaler = None

    def create_explainer(self, X, Y, normalize=True):
        if X is None or Y is None:
            raise ValueError("Both X and Y must be provided.")
        self.A_dagger, self.scaler, X_prime = self._generate_inverse_operator_from_y(X, Y, normalize)
        return self

    def _generate_inverse_operator_from_y(self, X, Y, normalize=True):
        if normalize:
            scaler = StandardScaler()
            X_prime = scaler.fit_transform(X)
        else:
            scaler = None
            X_prime = X

        y_pinv = np.linalg.pinv(Y)
        A_dagger = np.dot(y_pinv, X_prime).T

        return A_dagger, scaler, X_prime  # X_primeも返す

    def global_feature_importance(self, feature_names=None, class_names=None, top_k=None, top_k_criterion='average'):
        dim = self.A_dagger.shape[1]
        data = []
        for t in range(dim):
            basis = np.zeros(dim)
            basis[t] = 1
            heatmap = np.dot(self.A_dagger, basis)
            heatmap /= np.max(np.abs(heatmap))
            data.append(heatmap)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.A_dagger.shape[0])]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(dim)]

        df = pd.DataFrame(np.array(data), index=class_names, columns=feature_names)

        if top_k is not None:
            if top_k_criterion == 'average':
                top_k_features = df.mean(axis=0).nlargest(top_k).index.tolist()
            elif top_k_criterion == 'max':
                top_k_features = df.max(axis=0).nlargest(top_k).index.tolist()
            else:
                raise ValueError(f"Unknown top_k_criterion: {top_k_criterion}")
            df = df.loc[:, top_k_features]

        df_melted = df.reset_index().melt(id_vars='index', value_name='values', var_name='feature')
        df_melted = df_melted[~pd.to_numeric(df_melted['values'], errors='coerce').isnull()]

        plt.figure(figsize=(10, 8))
        sns.barplot(data=df_melted, x='values', y='feature', hue='index', palette='pastel', dodge=True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

        plt.figure(figsize=(10,6))
        sns.heatmap(df, cmap="Blues", annot=True, fmt=".2f")
        plt.show()

        return df

    def global_feature_importance_each(self, feature_names=None, class_names=None, top_k=None, top_k_criterion='average', class_num=0):
        dim = self.A_dagger.shape[1]
        data = []
        for t in range(dim):
            basis = np.zeros(dim)
            basis[t] = 1
            heatmap = np.dot(self.A_dagger, basis)
            heatmap /= np.max(np.abs(heatmap))
            data.append(heatmap)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.A_dagger.shape[0])]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(dim)]

        df = pd.DataFrame(np.array(data), index=class_names, columns=feature_names)
        df =df[class_num:class_num+1]

        if top_k is not None:
            if top_k_criterion == 'average':
                top_k_features = df.mean(axis=0).nlargest(top_k).index.tolist()
            elif top_k_criterion == 'max':
                top_k_features = df.max(axis=0).nlargest(top_k).index.tolist()
            else:
                raise ValueError(f"Unknown top_k_criterion: {top_k_criterion}")
            df = df.loc[:, top_k_features]

        df_melted = df.reset_index().melt(id_vars='index', value_name='values', var_name='feature')
        df_melted = df_melted[~pd.to_numeric(df_melted['values'], errors='coerce').isnull()]

        plt.figure(figsize=(10, 8))
        sns.barplot(data=df_melted, x='values', y='feature', hue='index', palette='pastel', dodge=True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

        plt.figure(figsize=(10,6))
        sns.heatmap(df, cmap="Blues", annot=True, fmt=".2f")
        plt.show()

        return df


    def global_feature_importance_without_viz(self, feature_names=None, class_names=None, top_k=None, top_k_criterion='average'):
        if self.A_dagger is None:
            raise ValueError("Please create an explainer first using create_explainer.")
        dim = self.A_dagger.shape[1]
        data = []
        for t in range(dim):
            basis = np.zeros(dim)
            basis[t] = 1
            heatmap = np.dot(self.A_dagger, basis)
            heatmap /= np.max(np.abs(heatmap))
            data.append(heatmap)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.A_dagger.shape[0])]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(dim)]

        df = pd.DataFrame(np.array(data), index=class_names, columns=feature_names)

        if top_k is not None:
            if top_k_criterion == 'average':
                top_k_features = df.mean(axis=0).nlargest(top_k).index.tolist()
            elif top_k_criterion == 'max':
                top_k_features = df.max(axis=0).nlargest(top_k).index.tolist()
            else:
                raise ValueError(f"Unknown top_k_criterion: {top_k_criterion}")
            df = df.loc[:, top_k_features]

        return df

