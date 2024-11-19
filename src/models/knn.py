from sklearn.neighbors import KNeighborsClassifier

# Retorna um classificador K-Nearest Neighbors com o número de vizinhos configurável
def criar_knn(n_neighbors=3):

    return KNeighborsClassifier(n_neighbors=n_neighbors)
