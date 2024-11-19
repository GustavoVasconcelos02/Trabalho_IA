from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# Avalia o modelo utilizando validação cruzada estratificada
def avaliar_modelo(modelo, X, y, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits)
    acuracias = cross_val_score(modelo, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(modelo, X, y, cv=skf, scoring='f1_weighted')
    
    return {
        'acuracia_media': acuracias.mean(),
        'acuracia_desvio': acuracias.std(),
        'f1_media': f1_scores.mean(),
        'f1_desvio': f1_scores.std()
    }
