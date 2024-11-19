from sklearn.neural_network import MLPClassifier

# Retorna um classificador de Rede Neural MLP
def criar_mlp(hidden_layer_sizes=(100,), max_iter=300, random_state=42):

    return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
