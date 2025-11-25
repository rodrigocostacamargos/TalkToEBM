# TalkToEBM Web Interface - Análise de Evasão UPE

Uma aplicação web Flask que fornece uma interface interativa para análise de modelos Explainable Boosting Machines (EBM) com linguagem natural.

## 📋 Funcionalidades

- **Interface Web Interativa**: Interface moderna e responsiva para interagir com modelos EBM
- **Análise de Features Individuais**: Clique em qualquer feature para obter uma descrição detalhada do LLM
- **Prompts Customizados**: Permite enviar instruções personalizadas para o LLM
- **Análise Completa do Modelo**: Resumo de todo o modelo EBM com insights do LLM
- **Suporte a Múltiplos Idiomas**: Português, Inglês e Espanhol
- **Exemplos de Prompts**: Botões rápidos para análises específicas
- **Cópia de Resultados**: Funcionalidade de copiar resultados para a área de transferência

## 🚀 Como Executar

### Pré-requisitos
- Python 3.8+
- Dependências listadas em `requirements.txt`

### Instalação

1. **Clone o repositório** (se aplicável):
```bash
git clone <seu-repositorio>
cd TalkToEBM
```

2. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

3. **Execute a aplicação**:
```bash
python app.py
```

4. **Acesse no navegador**:
```
http://localhost:5000
```

## 📁 Estrutura do Projeto

```
├── app.py                 # Aplicação Flask principal
├── evasao_upe.py          # Script Python convertido do notebook
├── requirements.txt       # Dependências do projeto
├── templates/
│   └── index.html        # Interface web
└── notebooks/
    └── EvasaoUPE.ipynb   # Notebook original
```

## 🔧 API Endpoints

### GET `/`
- **Descrição**: Página principal da aplicação web
- **Resposta**: Interface HTML interativa

### POST `/api/describe_graph`
- **Descrição**: Analisa um gráfico específico do modelo EBM
- **Parâmetros**:
  - `feature_index` (int): Índice da feature a ser analisada
  - `custom_prompt` (string, opcional): Prompt personalizado
  - `language` (string, opcional): Idioma da resposta
- **Resposta**: Descrição do gráfico pelo LLM

### POST `/api/describe_model`
- **Descrição**: Analisa o modelo EBM completo
- **Parâmetros**:
  - `custom_prompt` (string, opcional): Prompt personalizado
  - `language` (string, opcional): Idioma da resposta
- **Resposta**: Resumo completo do modelo pelo LLM

### GET `/api/features`
- **Descrição**: Lista todas as features disponíveis no modelo
- **Resposta**: Array de objetos com informações das features

### GET `/api/health`
- **Descrição**: Verifica o status da aplicação
- **Resposta**: Status do sistema e do modelo carregado

## 🎯 Exemplos de Uso

### Análise de Feature Individual
1. Na interface web, clique em qualquer feature na lista
2. O LLM irá descrever os padrões encontrados nessa feature
3. Use prompts customizados para focar em aspectos específicos

### Análise Completa do Modelo
1. Digite um prompt personalizado (opcional)
2. Clique em "Analisar Modelo Completo"
3. Receba um resumo abrangente de todas as features

### Prompts Sugeridos
- **🎯 Padrões Surpreendentes**: Identifica comportamentos inesperados
- **👨‍💼 Para Gestores**: Explicação em linguagem de negócios
- **🚀 Ações Práticas**: Sugere intervenções baseadas na análise
- **🔍 Análise Crítica**: Identifica limitações e vieses

## 🔍 Tecnologias Utilizadas

- **Flask**: Framework web para Python
- **Bootstrap 5**: Interface responsiva e moderna
- **TalkToEBM**: Integração com modelos EBM e LLMs
- **Explainable Boosting Machines**: Modelos interpretáveis de ML
- **JavaScript/Fetch API**: Comunicação assíncrona com o backend

## 📊 Dados e Modelo

A aplicação utiliza dados de evasão estudantil da UPE, incluindo:
- Cursos (Pedagogia, Licenciatura em Computação)
- Métricas de desempenho acadêmico (var01, var02, etc.)
- Informações de semestre e período
- Variável alvo: EVASAO (binária)

## 🛠️ Personalização

### Adicionar Novos Prompts
Edite a seção "Exemplos de Prompts" no `templates/index.html` para incluir novos templates.

### Modificar o Modelo
Substitua o arquivo `notebooks/dados/ebm_upe.joblib` por um novo modelo EBM treinado.

### Customizar a Interface
Modifique o arquivo `templates/index.html` para alterar o layout e funcionalidades.

## 📝 Notas Importantes

- A aplicação requer que o modelo EBM esteja previamente treinado
- As respostas do LLM dependem da qualidade e contexto dos dados
- Para produção, configure um servidor WSGI como Gunicorn
- Considere adicionar autenticação para ambientes de produção

## 🐛 Solução de Problemas

### Modelo Não Carregado
- Verifique se o arquivo `notebooks/dados/ebm_upe.joblib` existe
- Execute `evasao_upe.py` para treinar o modelo se necessário

### Erros de Importação
- Confirme que todas as dependências estão instaladas
- Verifique as versões das bibliotecas no `requirements.txt`

### Aplicação Não Inicia
- Verifique se a porta 5000 está disponível
- Confirme que o Python 3.8+ está instalado
