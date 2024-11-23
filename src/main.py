from sklearn.model_selection import train_test_split
from utils.pdf_utils import carregar_pdfs_de_diretorios
from utils.text_preprocessing import limpar_lista_textos
from utils.vectorizer import criar_matriz_bow

# Diretórios dos PDFs
diretorios = {
    'poesia': 'src/pdfs/poesia/',
    'prosa': 'src/pdfs/prosa/',
    'jornalismo': 'src/pdfs/jornalismo/'
}

# 1. Carregar textos e classes
textos, classes = carregar_pdfs_de_diretorios(diretorios)

# 2. Pré-processar os textos
textos_limpos = limpar_lista_textos(textos)

# 3. Criar a matriz Bag of Words
vectorizer, X = criar_matriz_bow(textos_limpos)

# 4. Exibir informações
print("Matriz BoW: ", X.toarray())
print("Vocabulário: ", vectorizer.get_feature_names_out())
print("Classes: ", classes)

# 5. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.3, random_state=42)
print("Dados divididos em treino e teste com sucesso!")
