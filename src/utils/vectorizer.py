from sklearn.feature_extraction.text import TfidfVectorizer

# Cria uma matriz Bag of Words (BoW) a partir de uma lista de textos
def criar_matriz_bow(textos):
    vectorizer = TfidfVectorizer()
    matriz_bow = vectorizer.fit_transform(textos)
    return vectorizer, matriz_bow
