# -*- coding: utf-8 -*-
"""
Script Python baseado no notebook EvasaoUPE.ipynb
Análise de evasão estudantil na UPE usando Explainable Boosting Machines (EBM)
e interface de linguagem natural com TalkToEBM
"""

import pandas as pd
import textwrap
from sklearn.model_selection import train_test_split

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

import t2ebm
import joblib
from pathlib import Path

# =============================================================================
# Carregue o dataset evasao_UPE.csv e treine um EBM
# =============================================================================

print("Carregando dados de evasão da UPE...")
df = pd.read_csv("notebooks/dados/evasao_UPE.csv", sep=';', decimal=',')

# Colunas de identificação e a probabilidade pré-calculada são removidas para evitar vazamento do alvo
cols_to_drop = ["Aluno", "ID do Aluno", "Disciplina", "ID da Disciplina", "PROBABILIDADE"]
df_model = df.drop(columns=cols_to_drop)

# pandas to numpy array
X_data = df_model.drop(columns=["EVASAO"]).values
y_data = df_model["EVASAO"].values

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

print("Primeiras linhas do dataset:")
print(df_model.head())

feature_names = df_model.drop(columns=["EVASAO"]).columns.tolist()

model_path = Path("notebooks/dados/ebm_upe.joblib")
model_loaded = model_path.exists()

if model_loaded:
    ebm = joblib.load(model_path)
    print(f"Modelo carregado de {model_path}")
else:
    print("Treinando novo modelo EBM...")
    ebm = ExplainableBoostingClassifier(interactions=0, 
                                        feature_names=feature_names)
    ebm.fit(X_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ebm, model_path)
    print(f"Modelo treinado e salvo em {model_path}")

ebm_score = ebm.score(X_test, y_test)
print(f"Acurácia do modelo EBM: {ebm_score:.4f}")

# =============================================================================
# Configurações para uso com LLM
# =============================================================================

# Descrições para uso com LLM
dataset_description = """Este notebook usa um conjunto de dados de evasão estudantil da UPE. Cada linha representa o histórico de um(a) aluno(a) em uma disciplina em um semestre específico. As colunas incluem o curso (Pedagogia ou Licenciatura em Computação), semestre/período, datas de início e finalização, e variáveis numéricas (var01, var02, var03, var10, var11, var12, var13, var16, var17, var18, var19, var20, var21, var26, var27, var29) com métricas derivadas do desempenho acadêmico e engajamento. As colunas de identificação e a probabilidade pré-computada foram removidas antes do treino para evitar vazamento. O alvo EVASAO é binário: 1 indica que o(a) aluno(a) abandonou, e 0 indica que permaneceu ativo(a)."""

y_axis_description = "O eixo y mostra as contribuições em log-odds para a chance de evasão (probabilidade de EVASAO = 1)."

print("\n" + "="*80)
print("CONFIGURAÇÃO CONCLUÍDA")
print("="*80)
print("O script foi executado com sucesso!")
print(f"- Modelo EBM treinado/carregado com acurácia: {ebm_score:.4f}")
print(f"- {len(feature_names)} features disponíveis para análise")
print("- Pronto para uso com interface web")
