import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

# Baixar recursos do NLTK (apenas na primeira execução)
nltk.download('punkt')
nltk.download('stopwords')

#Função para extrair texto de PDFs
def pdf_para_txt(caminho_pdf):
    with open(caminho_pdf, 'rb') as f:
        leitor = PyPDF2.PdfReader(f)
        texto = ""
        for pagina in range(len(leitor.pages)):
            texto += leitor.pages[pagina].extract_text() or ""
    return texto

#Função para limpar e remover stopwords
def limpar_texto(texto):
    stop_words = set(stopwords.words('portuguese'))
    palavras = word_tokenize(texto.lower())
    palavras_limpa = [palavra for palavra in palavras if palavra.isalnum() and palavra not in stop_words]
    return " ".join(palavras_limpa)

#Diretórios com os PDFs
diretorios = {
    'poesia': 'pdfs/poesia/',
    'prosa': 'pdfs/prosa/',
    'jornalismo': 'pdfs/jornalismo/'
}

#Extraindo textos e gerando classes
textos = []
classes = []
for classe, caminho in diretorios.items():
    for arquivo in os.listdir(caminho):
        if arquivo.endswith('.pdf'):
            texto = pdf_para_txt(os.path.join(caminho, arquivo))
            texto_limpo = limpar_texto(texto)
            textos.append(texto_limpo)
            classes.append(classe)

#Criando a matriz Bag of Words
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos)

#Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.3, random_state=42)

#Lista de classificadores
classificadores = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('KNN', KNeighborsClassifier()),
    ('Naive Bayes', MultinomialNB()),
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('MLP', MLPClassifier(max_iter=1000))
]

#Validação cruzada estratificada com 10 folds
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

#Avaliação dos classificadores
for nome, modelo in classificadores:
    scores_acuracia = cross_val_score(modelo, X_train, y_train, cv=kfold, scoring='accuracy')
    scores_f1 = cross_val_score(modelo, X_train, y_train, cv=kfold, scoring='f1_weighted')
    
    print(f"\nModelo: {nome}")
    print(f"Acurácia média: {scores_acuracia.mean():.4f} (+/- {scores_acuracia.std():.4f})")
    print(f"F1-Score médio: {scores_f1.mean():.4f} (+/- {scores_f1.std():.4f})")

#Otimização de hiperparâmetros com GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid_dt = {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}
grid_search_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=kfold, scoring='accuracy')
grid_search_dt.fit(X_train, y_train)

print("\nMelhores parâmetros para Decision Tree:", grid_search_dt.best_params_)
print("Melhor acurácia obtida:", grid_search_dt.best_score_)
