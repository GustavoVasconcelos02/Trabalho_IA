from sklearn.tree import DecisionTreeClassifier

# Retorna um classificador de Árvore de Decisão com hiperparâmetros configuráveis
def criar_decision_tree(max_depth=None, random_state=42):

    return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
