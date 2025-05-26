import numpy as np # linear algebrafrom re import T
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
df_gender = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")


print(train_df.isnull().sum())
print("\n\n")
print(test_df.isnull().sum())
print(train_df.describe())


#Gera o grafico de sobreviventes e mortos do train_df 
sns.countplot(x='Survived', data = train_df)
plt.title("Sobreviventes")
plt.show()
print("\n")
#Gera o grafico de mulheres e Homems do train_df 
sns.countplot(x='Sex', data = train_df)
plt.title("Sexo")
plt.show()
print("\n")
#Gera o grafico de quantidades de Classes ocupadas do train_df 
sns.countplot(x='Pclass', data = train_df)
plt.title("Classes")
plt.show()
print("\n")

#Colona a idade media(21.5) para linhas de idade nullas
train_df['Age'] = train_df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
test_df['Age'] = test_df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

#Coloca a Moda de embraques nas linhas de embarque se encontra nullos
moda_Embarked = train_df['Embarked'].mode()[0]
train_df = train_df.fillna(moda_Embarked)

#Coloca a mediana nas linhas de tarifas que estão nullas
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

#Pega apenas linhas não nullas
train_df['Has_Cabin'] = train_df['Cabin'].notnull().astype(int)
test_df['Has_Cabin'] = test_df['Cabin'].notnull().astype(int)

#Transforma o sexo em 0 e 1 em pro de melhor performace
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male':0, 'famale': 1})


test_df = pd.get_dummies(test_df, columns = ['Embarked'], prefix =  "Emb")
train_df = pd.get_dummies(train_df, columns=['Embarked'], prefix='Emb')
train_df["family_size"] = train_df['SibSp'] + train_df['Parch'] +1
test_df['family_size'] = test_df['SibSp'] + test_df['Parch'] + 1
train_df.head(50)

#Caracteristicas para treinar o modelo
#Definimos uma lista com os nomes das colunas 
features = ['Pclass', 'Sex', 'Age', 'Fare', 'family_size', 'Has_Cabin']

#Variavel independente contendo as colunas de carateristicas(Entrada)
x_train = train_df[features]

#Isolamos a coluna Sobreviente do train_df e atribuimos ela a y_train
y_train = train_df['Survived']

#Sub-Conjunto do test_df, uma lista com o nome das colunas
x_test = test_df[features]

#Padronização de valores
scaler = StandardScaler()
#Fit calcula a meida e o desvio padrão
#Transform aplica o Fit
x_train.loc[:, ['Age', 'Fare']] = scaler.fit_transform(x_train[['Age', 'Fare']])
#Aplica a padronização nas tabelas selecionadas 
x_test.loc[:, ['Age', 'Fare']] = scaler.transform(x_test[['Age', 'Fare']])

model = XGBClassifier(n_estimators = 49, max_depth = 4, random_state=77044)

model.fit(x_train,y_train)

scores = cross_val_score(model, x_train, y_train, cv=11, scoring='accuracy')
print(f"Acurácia média (validação cruzada Xbost): {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")


param_grid = {
    'n_estimators': [48,49,50,51,52],
    'max_depth': [4,5,6,7,8]
}
grid_search = GridSearchCV(XGBClassifier(random_state=77044), param_grid, cv=11, scoring='accuracy')
grid_search.fit(x_train, y_train)
print(f"Melhores parâmetros: {grid_search.best_params_}")

# #Pegando o melhor cv
# cv_valuesE = [10,11,12,13,14,15]
# for cv in cv_valuesE:
#     scores = cross_val_score(model, x_train, y_train, cv=cv, scoring='accuracy')
#     print(f"Ensemble - Acurácia média: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

pred = model.predict(x_train)
accuracy = accuracy_score(y_train, pred)
print("Acuracia nova:", accuracy)
    
# Previsão e geração do arquivo de submissão
model.fit(x_train, y_train)  # Treina o modelo no conjunto completo de treino
y_test_pred = model.predict(x_test)  # Faz previsões no conjunto de teste
submission = pd.DataFrame({  # Cria um DataFrame com as previsões
    'PassengerId': test_df['PassengerId'],  # IDs dos passageiros do teste
    'Survived': y_test_pred                # Previsões de sobrevivência
})
submission.to_csv('submission.csv', index=False)  # Salva as previsões em um arquivo CSV
print("Arquivo 'submission.csv' gerado com sucesso!")