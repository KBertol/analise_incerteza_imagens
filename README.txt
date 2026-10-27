================================================================
  INICIAÇÃO CIENTÍFICA — README
  Caracterização da Incerteza Mediante Análise de
  Similaridade de Imagens Médicas (RX)
  por Meio de Ferramentas Clássicas
================================================================
  DEPENDÊNCIAS
----------------------------------------------------------------
Instalar antes de rodar qualquer script:

    pip install opencv-python pandas tqdm
    pip install scikit-image scikit-learn
    pip install scipy pandas numpy matplotlib tqdm

----------------------------------------------------------------
  ESTRUTURA DE PASTAS ESPERADA
----------------------------------------------------------------
  /
  ├── Base/
  │   ├── valid_normais_frontal/        (imagens Saudavel)
  │   ├── valid_Doentes_frontal/        (imagens Doente)
  │   └── valid_desconhecidos_frontal/  (imagens Incerto)
  │
  ├── padrao.py
  ├── extrair_features.py
  ├── extrair_glcm.py
  ├── similaridade.py
  ├── visualizar.py
  ├── random_forest.py
  └── analise_limiar.py

----------------------------------------------------------------
  ORDEM DE EXECUÇÃO
----------------------------------------------------------------

PASSO 1 — Gerar inventário da base de dados
    python padrao.py
    Gera: inventario_base_dados.csv

PASSO 2A — Extrair features LBP
    python extrair_features.py
    Gera: features_lbp.csv

PASSO 2B — Extrair features GLCM
    python extrair_glcm.py
    Gera: features_glcm.csv

    Obs: 2A e 2B são independentes entre si.
         Ambos dependem do inventario_base_dados.csv (Passo 1).
         
PASSO OPCIONAL - Visualizar o ajuste por CLAHE nas imagens
    python visualizar_preprocessamento.py
    Imagem plotada na tela com o comparativo dos ajustes
    Obs: Depende do inventario_base_dados.csv (Passo 1).

PASSO 3 — Análise de similaridade cosseno + Kruskal-Wallis
    Editar similaridade.py: DESCRITOR = 'LBP'
    python similaridade.py
    Gera: similaridade_incertos.csv

    Editar similaridade.py: DESCRITOR = 'GLCM'
    python similaridade.py
    Gera: similaridade_incertos_glcm.csv

PASSO 4 — Visualizações (t-SNE + gráficos)
    Editar visualizar.py: DESCRITOR = 'LBP'
    python visualizar.py
    Gera: tsne_lbp.png
          histograma_lbp_classes.png
          similaridade_scatter_lbp.png

    Editar visualizar.py: DESCRITOR = 'GLCM'
    python visualizar.py
    Gera: tsne_glcm.png
          histograma_glcm_classes.png
          similaridade_scatter_glcm.png

    Obs: Passo 4 depende dos CSVs gerados no Passo 3.

PASSO 5 — Random Forest (LBP + GLCM combinados)
    python random_forest.py
    Gera: importancia_features.png
          matriz_confusao.png
          incertos_probabilidades.png
          incertos_incerteza.png
          incertos_probabilidades.csv

    Obs: Passo 5 depende de features_lbp.csv e features_glcm.csv.

PASSO 6 - Analise de Limiar
    python analise_limiar.py
    Gera: analise_limiar.csv 
          limiar_distribuicao_incertos.png
          limiar_desempenho.png
          limiar_histograma_incertos.png
          
    Obs: Passo 6 depende das features.png geradas no passo 2
    
----------------------------------------------------------------
  ARQUIVOS GERADOS (RESUMO)
----------------------------------------------------------------

  CSVs:
    inventario_base_dados.csv       Mapeamento das 4.003 imagens
    features_lbp.csv                Features LBP (4.002 x 26 bins)
    features_glcm.csv               Features GLCM (4.002 x 6 features)
    similaridade_incertos.csv       Similaridade cosseno LBP
    similaridade_incertos_glcm.csv  Similaridade cosseno GLCM
    incertos_probabilidades.csv     Probabilidades RF para incertos
    analise_limiar.csv              Análise do limiar (0.1 - 0.9)

  Gráficos LBP:
    tsne_lbp.png                    Projeção t-SNE das features LBP
    histograma_lbp_classes.png      Histograma LBP médio por classe
    similaridade_scatter_lbp.png    Scatter similaridade cosseno LBP

  Gráficos GLCM:
    tsne_glcm.png                   Projeção t-SNE das features GLCM
    histograma_glcm_classes.png     Features GLCM médias por classe
    similaridade_scatter_glcm.png   Scatter similaridade cosseno GLCM

  Gráficos Random Forest:
    importancia_features.png          Ranking de importância Gini (top 15)
    matriz_confusao_rand_forest.png   Matriz de confusão no conjunto teste
    incertos_probabilidades.png       Scatter P(Saudavel) x P(Doente) dos incertos
    incertos_incerteza.png            Histograma de incerteza |P_D - P_S|

  Gráficos Análise de Limiar:
    limiar_distribuicao_incertos.png  Classificação por Limiar de Decisão
    limiar_desempenho.png             Desempenho por Limiar de Decisão
    limiar_histograma_incertos.png    Distribuição P(Doente) para os incertos


