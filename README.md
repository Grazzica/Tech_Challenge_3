# Tech Challenge 3 — Previsão de Atrasos em Voos

> Pipeline de Machine Learning para prever se um voo chegará atrasado, utilizando dados públicos de voos dos EUA (2015). Projeto desenvolvido para a pós-graduação em Machine Learning Engineering (FIAP).

---

## Índice

- [Objetivo](#objetivo)
- [Dados](#dados)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Fluxo de Dados](#fluxo-de-dados)
- [Estrutura de Pastas](#estrutura-de-pastas)
- [Como Executar](#como-executar)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [Limitações e Próximos Passos](#limitações-e-próximos-passos)
- [Tecnologias](#tecnologias)
- [Deploy](#deploy)
- [Autores](#autores)

---

## Objetivo

Construir um pipeline completo de ciência de dados — da exploração à interpretação dos resultados — aplicando técnicas de Machine Learning **supervisionado** e **não supervisionado**.

O problema de negócio: prever se um voo chegará atrasado (atraso ≥ 15 min na chegada, padrão FAA) **antes da partida**, permitindo ação antecipada (renegociação de passagens, priorização operacional).

> **Decisão metodológica:** apenas features disponíveis *antes do voo* são utilizadas. Variáveis conhecidas só após a ocorrência do voo foram descartadas para evitar *data leakage*.

---

## Dados

Base pública de voos domésticos dos EUA (2015), composta por três tabelas:

| Tabela | Descrição | Registros |
|:---|:---|:---|
| `flights.csv` | Base principal — um registro por voo | ~5,8M |
| `airports.csv` | Dicionário de aeroportos (código IATA, cidade, estado, coordenadas) | 322 |
| `airlines.csv` | Dicionário de companhias aéreas | 14 |

**Target:** `ARRIVAL_DELAY_CLASS` — binário, derivado de `ARRIVAL_DELAY ≥ 15 min`.
**Distribuição:** ~81% não atrasados / ~19% atrasados (classes desbalanceadas).

---

## Arquitetura do Projeto

```mermaid
flowchart TD
    A[flights.csv<br/>airports.csv<br/>airlines.csv] --> B[01_eda.ipynb<br/>Análise Exploratória]
    B -->|define decisões de<br/>pré-processamento| C[data_loader.py<br/>Pipeline reproduzível]
    A --> C
    C --> D[(flights_processed.parquet<br/>~5,2M linhas)]
    D --> E[02_class_model_supervised.ipynb<br/>Classificação]
    A --> F[03_class_model_unsupervised.ipynb<br/>Clusterização de aeroportos]
    E --> G[Comparação de Modelos<br/>+ Apresentação Crítica]
    F --> H[Perfis de Aeroportos<br/>+ Apresentação Crítica]
```

> O fluxo reflete a ordem real do desenvolvimento: a **análise exploratória** (`01_eda.ipynb`) foi realizada primeiro e definiu as decisões de limpeza, seleção de features e tratamento de leakage. Essas decisões foram implementadas no `data_loader.py`, que gera o dataset processado consumido pela **classificação** (`02`). A **clusterização** (`03`) parte do CSV bruto, pois utiliza features distintas — incluindo variáveis pós-voo (causas de atraso) que seriam leakage na classificação, mas são válidas em uma análise descritiva.

---

## Fluxo de Dados

Etapas de transformação aplicadas pelo `data_loader.py`, na ordem de execução:

```mermaid
flowchart LR
    A[CSV bruto] --> B[Remover voos<br/>cancelados/desviados]
    B --> C[Remover linhas<br/>SCHEDULED_TIME nulo]
    C --> D[Remover códigos<br/>de aeroporto inválidos]
    D --> E[Decompor horário<br/>em HORA + MINUTO]
    E --> F[Criar target<br/>binário]
    F --> G[Remover colunas<br/>não utilizadas / leakage]
    G --> H[(Parquet<br/>processado)]
```

**Funil de remoção de linhas:** voos cancelados/desviados, registros com `SCHEDULED_TIME` nulo e códigos de aeroporto em formato inválido são removidos, reduzindo a base de ~5,8M para ~5,2M de linhas. O detalhamento por etapa está documentado no notebook `01_eda.ipynb`.

---

## Estrutura de Pastas

```
Tech_challenge_3/
├── data/
│   ├── raw/                 # Dados brutos (imutáveis)
│   │   ├── flights.csv
│   │   ├── airports.csv
│   │   └── airlines.csv
│   └── processed/           # Dados processados (parquet)
├── notebooks/
│   ├── 01_eda.ipynb                       # Análise exploratória
│   ├── 02_class_model_supervised.ipynb    # Modelagem supervisionada (classificação)
│   └── 03_class_model_unsupervised.ipynb  # Modelagem não supervisionada (clusterização)
├── src/
│   ├── __init__.py
│   └── data_loader.py       # Pipeline de pré-processamento
├── reports/                 # Gráficos e artefatos de apresentação
├── pyproject.toml           # Configuração do pacote
├── .gitignore
└── README.md
```

---

## Como Executar

> **Pré-requisitos:** Python ≥ 3.10. Recomendado ambiente virtual.

```bash
# 1. Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Linux/WSL/Mac
# .venv\Scripts\activate           # Windows

# 2. Instalar dependências e o pacote local em modo editável
pip install -e .

# 3. Adicionar os dados brutos em data/raw/
#    (flights.csv, airports.csv, airlines.csv)

# 4. Executar os notebooks na ordem
#    notebooks/01_eda.ipynb
#    notebooks/02_class_model_supervised.ipynb
#    notebooks/03_class_model_unsupervised.ipynb
```

> O `02_class_model_supervised.ipynb` gera o parquet processado na primeira execução
> (via `data_loader.load_flights`). Execuções seguintes podem ler o parquet
> diretamente para maior velocidade. O `03_class_model_unsupervised.ipynb` parte
> diretamente do `flights.csv`, pois usa um conjunto de features próprio.

---

## Metodologia

### Análise Exploratória (`01_eda.ipynb`)
- Análise de valores faltantes com verificação de hipóteses
- Identificação e tratamento de *data leakage*
- Seleção de features com base na relação com o target
- Definição das decisões de pré-processamento

### Modelagem Supervisionada (`02_class_model_supervised.ipynb`)
- **Baseline:** `DummyClassifier` (referência mínima)
- **Modelos comparados:** Decision Tree e Random Forest
- **Seleção de hiperparâmetro:** curva de validação (método do cotovelo para *max_depth*)
- **Tratamento de desbalanceamento:** `class_weight='balanced'`
- **Avaliação:** precision, recall e F1 da classe de interesse (não apenas acurácia)

### Modelagem Não Supervisionada (`03_class_model_unsupervised.ipynb`)
- **Técnica:** K-Means (clusterização de aeroportos por perfil operacional)
- **Unidade de análise:** aeroporto de origem (voos agregados por `ORIGIN_AIRPORT`)
- **Features (9):** volume de voos, taxa de atraso, atraso médio, taxa de cancelamento, tempo médio de táxi e decomposição das causas de atraso (tráfego aéreo, companhia, clima, efeito cascata)
- **Pré-processamento:** padronização (StandardScaler); aeroportos com menos de 1.000 voos descartados
- **Seleção de K:** método do cotovelo
- **Visualização:** PCA (redução para 2 dimensões)

---

## Resultados

### Classificação

Entre os modelos testados, o **Random Forest balanceado** apresentou o melhor resultado para uso real, com o melhor F1 (0,45) na classe de interesse — ou seja, o melhor equilíbrio entre capturar atrasos reais (recall) e acertar quando prevê atraso (precision).

A superioridade esperada da Random Forest sobre a Decision Tree só se confirmou após o tratamento de desbalanceamento: nos modelos sem balanceamento, ambas tiveram F1 equivalente (~0,17–0,18), pois o desbalanceamento limitava as duas igualmente. O tratamento com `class_weight='balanced'` foi o fator de maior impacto, elevando a capacidade de detectar atrasos.

Apesar de ser o melhor modelo, o F1 de 0,45 indica desempenho limitado em termos absolutos — o modelo captura pouco mais da metade dos atrasos, apontando para um teto imposto pelas features disponíveis.

### Clusterização (perfis de aeroportos)

O K-Means agrupou 225 aeroportos (com volume ≥ 1.000 voos) em **4 perfis distintos**. O achado mais relevante é que os grupos não se separam apenas por tamanho, mas principalmente pela **causa dominante do atraso**:

| Cluster | Perfil | Característica principal |
|:---|:---|:---|
| Eficientes | Porte médio, melhor desempenho | Menor taxa de atraso e menor atraso médio (ex: SAN, TPA, BNA) |
| Mega Hubs | Volume gigantesco | Atrasos ligados à própria companhia (ex: ATL, ORD, DFW, DEN, LAX) |
| Alto Atraso | Pior desempenho do ano | Maior atraso médio e maior taxa de cancelamento (ex: MDW, DAL, HOU) |
| Sensíveis a Tráfego Aéreo | Gargalo de espaço aéreo | Causa dominante é o controle de tráfego (ex: DCA, CLE, RDU, MKE) |

O PCA foi utilizado para projetar os 9 atributos em 2 dimensões e visualizar os agrupamentos, preservando cerca de 49% da variância.

**As duas análises se complementam:** a clusterização confirma que a feature `ORIGIN_AIRPORT` é relevante para a classificação, ao revelar que existem aeroportos estruturalmente mais propensos a atraso — e que a natureza desse atraso varia conforme o perfil do aeroporto.

---

## Limitações e Próximos Passos

### Limitações
- **Teto de features (classificação):** variáveis pré-voo não capturam as causas reais do atraso (clima, falhas mecânicas, efeito cascata). Os algoritmos testados atingiram limite de desempenho semelhante.
- **Desempenho absoluto:** F1 de 0,45 — o modelo captura ~54% dos atrasos, insuficiente para produção sem melhorias.
- **Recursos computacionais:** RAM limitou os hiperparâmetros da Random Forest (50 árvores, profundidade 20).
- **Cobertura da clusterização:** o dataset usa dois formatos de código de aeroporto (IATA e numérico do BTS); como o `airports.csv` cobre apenas o IATA, parte dos registros foi descartada na análise não supervisionada.
- **Escolha de K (clusterização):** o método do cotovelo não apresentou um ponto de inflexão nítido; K=4 foi escolhido com peso na interpretabilidade dos perfis.

### Próximos Passos
- Enriquecer a base com features de maior impacto: clima, congestionamento por horário, feriados, efeito cascata da aeronave.
- Incorporar dados geográficos (latitude/longitude) dos aeroportos.
- Com mais recursos computacionais, testar a Random Forest com mais árvores e maior profundidade, além de modelos mais robustos de maior demanda computacional.
- Aplicar K-Means a outras dimensões (companhias aéreas, rotas).
- Desenvolver modelo de regressão para prever a duração do atraso em minutos.

---

## Tecnologias

- **Python** 3.12
- **pandas** / **numpy** — manipulação de dados
- **scikit-learn** — modelagem (classificação, clusterização K-Means, PCA, métricas)
- **matplotlib** / **seaborn** — visualização
- **pyarrow** — armazenamento em Parquet
- **Jupyter** — notebooks de análise

---

## Deploy

- **Vídeo de apresentação disponível em:** https://youtu.be/Od04yPuuBuk

---

## Autores

**Adriano Cabrera**
- LinkedIn: https://www.linkedin.com/in/adriano-cabrera-b7b680a7/
- GitHub: https://github.com/cabrpin

**Caio Grazzini**
- LinkedIn: https://www.linkedin.com/in/caiograzzini/
- GitHub: https://github.com/Grazzica/

**Fabrício Batista Dias**
- LinkedIn: https://www.linkedin.com/in/fabriciobdias/
- GitHub: https://github.com/DiasFabricio

---

_Desenvolvido como parte do Tech Challenge — Pós-Graduação em Machine Learning Engineering — FIAP 2024/2025_
