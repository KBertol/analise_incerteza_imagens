# Caracterização da Incerteza em Imagens Médicas (RX) por Análise de Similaridade

> Iniciação Científica — UDESC · Ciência da Computação
> Caracterização da incerteza mediante análise de similaridade de imagens médicas de raio-X por meio de **ferramentas clássicas** de visão computacional e aprendizado de máquina.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-CV-5C3EE8?logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/status-em%20andamento-yellow)

---

## 📌 Sobre o projeto

A base de dados é composta por radiografias frontais divididas em três classes:

| Classe | Pasta | Descrição |
|---|---|---|
| **Saudavel** | `valid_normais_frontal` | Imagens normais |
| **Doente** | `valid_Doentes_frontal` | Imagens com patologia |
| **Incerto** | `valid_desconhecidos_frontal` | Imagens de classificação duvidosa |

**Hipótese:** as imagens *Incertas* são incertas porque têm características parecidas com **ambas** as classes (Saudável e Doente), e não por ruído ou falha de aquisição.

Para investigar isso, o projeto:

1. Extrai descritores clássicos de textura (**LBP** e **GLCM**) após pré-processamento com **CLAHE**;
2. Mede a **similaridade cosseno** entre cada imagem incerta e o centroide de cada classe;
3. Testa a diferença entre as classes com **Kruskal-Wallis**;
4. Visualiza o espaço de features com **t-SNE**;
5. Treina classificadores (**Random Forest** e **Regressão Logística**) nas classes conhecidas e analisa as **probabilidades** atribuídas aos incertos;
6. Varre **limiares de decisão** para avaliar a sensibilidade da classificação.

---

## Metodologia

```mermaid
flowchart TD
    A[Base de imagens<br/>Saudavel / Doente / Incerto] --> B[padrao.py<br/>Inventário CSV]
    B --> C[Pré-processamento<br/>resize 512×512 + CLAHE]
    C --> D[extrair_features.py<br/>LBP]
    C --> E[extrair_glcm.py<br/>GLCM]
    D --> F[similaridade.py<br/>Cosseno + Kruskal-Wallis]
    E --> F
    F --> G[visualizar.py<br/>t-SNE e gráficos]
    D --> H[random_forest.py]
    E --> H
    D --> I[analise_incerteza.py<br/>Regressão Logística]
    E --> I
    H --> J[analise_limiar.py]
    I --> J
```

### Pré-processamento
- Leitura em escala de cinza e redimensionamento para **512 × 512** (`INTER_AREA`)
- **CLAHE** (`clipLimit=2.0`, `tileGridSize=(8, 8)`) para realçar o contraste local
- Mesmo pré-processamento para LBP e GLCM, permitindo comparação justa

### Descritores
| Descritor | Parâmetros | Nº de features |
|---|---|---|
| **LBP** | raio = 3, 24 pontos, método `uniform` | 26 bins (histograma normalizado) |
| **GLCM** | distância = 5, 4 ângulos (0°, 45°, 90°, 135°), média entre ângulos | 6 (contraste, dissimilaridade, homogeneidade, energia, correlação, entropia) |

### Modelos
| Modelo | Configuração |
|---|---|
| **Random Forest** | 200 árvores, `class_weight='balanced'`, split 80/20 estratificado, CV 5-fold |
| **Regressão Logística** | `lbfgs`, `C=1.0`, `class_weight='balanced'`, features padronizadas (`StandardScaler`), split 80/20 estratificado, CV 5-fold |

Os modelos são treinados **apenas com Saudável e Doente**; as imagens Incertas nunca são vistas no treino e são usadas apenas para medir a incerteza (probabilidades, margem e logit).

> Todos os scripts usam `SEED = 42` para garantir reprodutibilidade.

---

## 📁 Estrutura do repositório

```
.
├── Base/
│   ├── valid_normais_frontal/          # imagens Saudavel
│   ├── valid_Doentes_frontal/          # imagens Doente
│   └── valid_desconhecidos_frontal/    # imagens Incerto
│
├── padrao.py                           # inventário da base
├── extrair_features.py                 # features LBP
├── extrair_glcm.py                     # features GLCM
├── visualizar_preprocessamento.py      # (opcional) efeito do CLAHE
├── similaridade.py                     # similaridade cosseno + Kruskal-Wallis
├── visualizar.py                       # t-SNE e gráficos
├── random_forest.py                    # Random Forest (LBP + GLCM)
├── analise_incerteza.py                # Regressão Logística (softmax / logit)
├── analise_limiar.py                   # varredura de limiares
└── README.md
```

> ⚠️ A pasta `Base/` com as imagens não está incluída no repositório.

---

## ⚙️ Instalação

```bash
git clone <url-do-repositorio>
cd <nome-do-repositorio>

pip install opencv-python pandas numpy tqdm scikit-image scikit-learn scipy matplotlib seaborn
```

---

## Ordem de execução

### Passo 1 — Inventário da base
```bash
python padrao.py
```
**Gera:** `inventario_base_dados.csv`

### Passo 2 — Extração de features
Os passos 2A e 2B são independentes entre si, mas ambos dependem do passo 1.

```bash
python extrair_features.py   # 2A — LBP
python extrair_glcm.py       # 2B — GLCM
```
**Gera:** `features_lbp.csv` · `features_glcm.csv`

### Opcional — Visualizar o efeito do CLAHE
```bash
python visualizar_preprocessamento.py
```
Exibe na tela um comparativo (original × CLAHE) de uma amostra de cada classe. Depende do passo 1.

### Passo 3 — Similaridade cosseno + Kruskal-Wallis
Edite `DESCRITOR` no início de `similaridade.py` e rode uma vez para cada descritor:

```bash
# DESCRITOR = 'LBP'
python similaridade.py     # → similaridade_incertos.csv

# DESCRITOR = 'GLCM'
python similaridade.py     # → similaridade_incertos_glcm.csv
```

O script calcula a similaridade de cada imagem incerta com o centroide de Saudável e de Doente e interpreta a diferença média `|sim_saudavel − sim_doente|`:

| Diferença média | Interpretação |
|---|---|
| < 0,01 | Confirma a hipótese |
| < 0,02 | Sugere a hipótese, com variação |
| ≥ 0,02 | Não confirma de forma clara |

### Passo 4 — Visualizações (t-SNE + gráficos)
Depende dos CSVs do passo 3. Edite `DESCRITOR` em `visualizar.py` e rode para cada descritor:

```bash
python visualizar.py
```
**Gera (LBP):** `tsne_lbp.png` · `histograma_lbp_classes.png` · `similaridade_scatter_lbp.png`
**Gera (GLCM):** `tsne_glcm.png` · `histograma_glcm_classes.png` · `similaridade_scatter_glcm.png`

### Passo 5 — Random Forest (LBP + GLCM combinados)
Depende de `features_lbp.csv` e `features_glcm.csv`.

```bash
python random_forest.py
```
**Gera:** `importancia_features.png` · `matriz_confusao_rand_forest.png` · `incertos_probabilidades.png` · `incertos_incerteza.png` · `incertos_probabilidades.csv`

### Passo 6 — Regressão Logística (softmax e logit)
Depende de `features_lbp.csv` e `features_glcm.csv`.

```bash
python analise_incerteza.py
```
**Gera:** `incertos_softmax.csv` · `logit_matriz_confusao.png` · `softmax_distribuicao.png` · `logit_distribuicao.png`

Interpretação do logit: `logit ≈ 0` → máxima incerteza · `logit > 0` → pende para Doente · `logit < 0` → pende para Saudável.

### Passo 7 — Análise de limiar de decisão
Depende de `features_lbp.csv` e `features_glcm.csv`.

```bash
python analise_limiar.py
```
Varre limiares de 0,1 a 0,9 (passo 0,05) e mostra como a classificação dos incertos muda com o corte.

**Gera:** `analise_limiar.csv` · `limiar_distribuicao_incertos.png` · `limiar_desempenho.png` · `limiar_histograma_incertos.png`

---

## Arquivos gerados

<details>
<summary><b>CSVs</b></summary>

| Arquivo | Conteúdo |
|---|---|
| `inventario_base_dados.csv` | Mapeamento das 4.003 imagens |
| `features_lbp.csv` | Features LBP (4.002 × 26 bins) |
| `features_glcm.csv` | Features GLCM (4.002 × 6 features) |
| `similaridade_incertos.csv` | Similaridade cosseno (LBP) |
| `similaridade_incertos_glcm.csv` | Similaridade cosseno (GLCM) |
| `incertos_probabilidades.csv` | Probabilidades do Random Forest para os incertos |
| `incertos_softmax.csv` | Probabilidades, margem e logit (Regressão Logística) |
| `analise_limiar.csv` | Resultados da varredura de limiares |

</details>

<details>
<summary><b>Gráficos</b></summary>

| Grupo | Arquivo | Descrição |
|---|---|---|
| LBP | `tsne_lbp.png` | Projeção t-SNE das features LBP |
| LBP | `histograma_lbp_classes.png` | Histograma LBP médio por classe |
| LBP | `similaridade_scatter_lbp.png` | Scatter da similaridade cosseno (LBP) |
| GLCM | `tsne_glcm.png` | Projeção t-SNE das features GLCM |
| GLCM | `histograma_glcm_classes.png` | Features GLCM médias por classe |
| GLCM | `similaridade_scatter_glcm.png` | Scatter da similaridade cosseno (GLCM) |
| Random Forest | `importancia_features.png` | Ranking de importância Gini (top 15) |
| Random Forest | `matriz_confusao_rand_forest.png` | Matriz de confusão no conjunto de teste |
| Random Forest | `incertos_probabilidades.png` | Scatter P(Saudavel) × P(Doente) dos incertos |
| Random Forest | `incertos_incerteza.png` | Histograma de incerteza \|P_D − P_S\| |
| Reg. Logística | `logit_matriz_confusao.png` | Matriz de confusão no conjunto de teste |
| Reg. Logística | `softmax_distribuicao.png` | Distribuição de P(Doente) dos incertos |
| Reg. Logística | `logit_distribuicao.png` | Distribuição do logit dos incertos |
| Limiar | `limiar_distribuicao_incertos.png` | Classificação dos incertos por limiar |
| Limiar | `limiar_desempenho.png` | Acurácia e F1 por limiar |
| Limiar | `limiar_histograma_incertos.png` | Distribuição de P(Doente) para os incertos |

</details>

---

## 👩‍💻 Autora

**Karla Bertol** — Ciência da Computação, UDESC (Universidade do Estado de Santa Catarina)
