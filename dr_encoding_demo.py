import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    return X, y, feature_names, iris.target_names

# ---------- PCA ----------
def run_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_

# ---------- t-SNE ----------
def run_tsne(X, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

# ---------- UMAP ----------
def run_umap(X, n_components=2):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    X_umap = reducer.fit_transform(X)
    return X_umap

# ---------- Autoencoder ----------
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

def run_autoencoder(X, epochs=100, lr=1e-3):
    device = torch.device('cpu')
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model = Autoencoder(input_dim=X.shape[1], latent_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        X_recon, z = model(X_tensor)
        loss = criterion(X_recon, X_tensor)
        loss.backward()
        optimizer.step()
    return z.detach().numpy()

def run_encodings(y):
    # Label Encoding
    le = LabelEncoder()
    y_le = le.fit_transform(y)
    # One-Hot Encoding
    ohe = OneHotEncoder(sparse=False)
    y_ohe = ohe.fit_transform(y.reshape(-1, 1))
    return y_le, y_ohe

sentences = [['machine', 'learning'], ['dimensionality', 'reduction'], ['autoencoder', 'encoding']]
word2vec_model = Word2Vec(sentences, vector_size=4, min_count=1)

def main():
    st.title('show')
    X, y, features, labels = load_data()
    method = st.sidebar.selectbox('choose which one', ['PCA', 't-SNE', 'UMAP', 'Autoencoder'])
  
    if method == 'PCA':
        X_dr, var_ratio = run_pca(X)
        st.write(f'Explained variance ratio: {var_ratio}')
    elif method == 't-SNE':
        X_dr = run_tsne(X)
    elif method == 'UMAP':
        X_dr = run_umap(X)
    else:
        X_dr = run_autoencoder(X)
    
  df = pd.DataFrame(X_dr, columns=['dim1', 'dim2'])
    df['label'] = y
    fig, ax = plt.subplots()
    for lbl in np.unique(y):
        idx = df['label'] == lbl
        ax.scatter(df.loc[idx, 'dim1'], df.loc[idx, 'dim2'], label=labels[lbl])
    ax.legend()
    st.pyplot(fig)
    st.header('instance')
    y_le, y_ohe = run_encodings(y)
    st.write('Label Encoding:', y_le[:5])
    st.write('One-Hot Encoding (first 5 elements):', y_ohe[:5])
    # Embedding
    st.write('Word2Vec Embedding:')
    st.write('"machine"Vector:', word2vec_model.wv['machine'])

if __name__ == '__main__':
    main()
