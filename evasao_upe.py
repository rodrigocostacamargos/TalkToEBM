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

# Dicionário com descrições detalhadas de cada variável
feature_descriptions = {
    "Curso": "Curso do aluno (Pedagogia ou Licenciatura em Computação).",
    "Semestre": "Semestre letivo em que o aluno estava matriculado.",
    "Periodo": "Período do curso do aluno.",
    "Data de Início": "Data de início do semestre letivo.",
    "Data de Final": "Data de finalização do semestre letivo.",
    "var01": "Média semanal da quantidade de acessos do aluno ao ambiente no semestre.",
    "var02": "Quantidade de acessos do aluno ao ambiente por turno (Manhã), por semestre.",
    "var03": "Quantidade de acessos do aluno ao ambiente por turno (Tarde), por semestre.",
    "var04": "Quantidade de acessos do aluno ao ambiente por turno (Noite), por semestre.",
    "var05": "Quantidade de acessos do aluno ao ambiente por turno (Madrugada), por semestre.",
    "var06": "Tempo médio semanal de utilização da plataforma pelo aluno no semestre.",
    "var07": "Quantidade de acessos do aluno ao ambiente no semestre.",
    "var08": "Quantidade de diferentes locais ou momentos (IP's) a partir dos quais o aluno acessou o ambiente, por semestre.",
    "var09": "Quantidade de acessos do aluno aos diferentes tipos de recursos disponibilizados (página web, vídeo, pdfs, entre outros), por disciplina.",
    "var10": "Quantidade de acessos do aluno aos diferentes tipos de atividades disponibilizadas (webquest, forum, quiz, entre outros), por disciplina.",
    "var11": "Média semanal da quantidade de mensagens enviadas pelo aluno dentro do ambiente, por semestre.",
    "var12": "Quantidade de acessos do aluno aos fóruns, por disciplina.",
    "var13": "Quantidade geral de postagens do aluno em fóruns, por disciplina.",
    "var14": "Quantidade de postagens do aluno em fóruns que foram respondidas pelo professor ou tutor, por disciplina.",
    "var15": "Quantidade de postagens do aluno em fóruns que foram respondidas por outros alunos, por disciplina.",
    "var16": "Quantidade geral de mensagens enviadas pelo aluno dentro do ambiente, por semestre.",
    "var17": "Quantidade geral de mensagens recebidas pelo aluno dentro do ambiente, por semestre.",
    "var18": "Quantidade de colegas diferentes para quem o aluno enviou mensagens no ambiente, por semestre.",
    "var19": "Quantidade de mensagens dos professores recebidas pelo aluno no ambiente, por semestre.",
    "var20": "Quantidade de mensagens de colegas recebidas pelo aluno no ambiente, por semestre.",
    "var21": "Quantidade de mensagens enviadas pelo aluno para outros colegas no ambiente, por semestre.",
    "var22": "Quantidade de respostas de um professor para as dúvidas do aluno em fóruns, por disciplina.",
    "var23": "Quantidade de mensagens enviadas pelo aluno aos professores pelo ambiente, por semestre.",
    "var24": "Quantidade de tópicos criados pelo aluno em fórum do tipo 'tira-dúvidas' por disciplina.",
    "var25": "Quantidade de postagens do aluno em fóruns em resposta a outros alunos por disciplina.",
    "var26": "Quantidade geral de recursos disponibilizados pelo professor (página web, vídeo, pdfs, entre outros) por disciplina.",
    "var27": "Quantidade geral de atividades disponibilizadas (webquest, fórum, quiz, entre outros) pelo professor por disciplina.",
    "var28": "Quantidade de atividades com prazos de resposta ou envio definidos por professor, por disciplina.",
    "var29": "Quantidade de fóruns de discussão disponibilizados sobre os conteúdos por disciplina.",
    "var30": "Quantidade de sessões de web conferências disponibilizadas no curso, por disciplina.",
    "var31": "Disponibilidade (existência) de página com a agenda (cronograma) do curso ou disciplina.",
    "var32": "Quantidade de atividades entregues pelo aluno no prazo, por disciplina.",
    "var33": "Quantidade de atividades entregues pelo aluno fora do prazo, por disciplina.",
}

# Função para obter descrição de uma feature
def get_feature_description(feature_name):
    """Retorna a descrição detalhada de uma feature."""
    return feature_descriptions.get(feature_name, f"Variável {feature_name} (sem descrição disponível)")

# Gerar descrição detalhada das features usadas no modelo
def get_features_description_text():
    """Gera texto descritivo das features presentes no modelo."""
    descriptions = []
    for fname in feature_names:
        desc = feature_descriptions.get(fname, "Sem descrição disponível")
        descriptions.append(f"- {fname}: {desc}")
    return "\n".join(descriptions)

# Descrições para uso com LLM
dataset_description = f"""Este dataset contém dados de evasão estudantil da UPE (Universidade de Pernambuco) em cursos de educação a distância. Cada linha representa o histórico de um(a) aluno(a) em uma disciplina em um semestre específico.

**Contexto:** Os dados foram coletados de um Ambiente Virtual de Aprendizagem (AVA/Moodle) e incluem métricas de engajamento, participação e interação dos alunos.

**Variáveis do modelo:**
{get_features_description_text()}

**Alvo (EVASAO):** Variável binária onde 1 indica que o(a) aluno(a) abandonou o curso e 0 indica que permaneceu ativo(a).

**Observações importantes:**
- As variáveis medem diferentes aspectos do comportamento do aluno no ambiente virtual
- Métricas de acesso, mensagens e participação em fóruns são indicadores de engajamento
- Variáveis como var26, var27, var29 medem recursos oferecidos pelo professor (contexto do curso)"""

y_axis_description = "O eixo y mostra as contribuições em log-odds para a chance de evasão (probabilidade de EVASAO = 1)."

print("\n" + "="*80)
print("CONFIGURAÇÃO CONCLUÍDA")
print("="*80)
print("O script foi executado com sucesso!")
print(f"- Modelo EBM treinado/carregado com acurácia: {ebm_score:.4f}")
print(f"- {len(feature_names)} features disponíveis para análise")
print("- Pronto para uso com interface web")
