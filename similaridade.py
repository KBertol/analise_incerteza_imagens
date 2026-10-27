import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import kruskal

# 0. REPRODUTIBILIDADE
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 1. CONFIGURAÇÕES — altere aqui
# Opções: 'LBP' ou 'GLCM'
DESCRITOR = 'LBP'

CONFIGS = {
    'LBP': {
        'features_csv':  'features_lbp.csv',
        'output_csv':    'similaridade_incertos.csv',
        'prefixo_cols':  'lbp_bin_',   
        'label_cols':    'bins LBP',
    },
    'GLCM': {
        'features_csv':  'features_glcm.csv',
        'output_csv':    'similaridade_incertos_glcm.csv',
        'prefixo_cols':  None,          
        'label_cols':    'features GLCM',
    },
}

cfg     = CONFIGS[DESCRITOR]
excluir = ['nome_arquivo', 'classe', 'caminho']

# 2. CARREGAR DADOS
print("=" * 60)
print(f"  ANALISE DE SIMILARIDADE — {DESCRITOR}")
print("=" * 60)

df = pd.read_csv(cfg['features_csv'])

if cfg['prefixo_cols']:
    cols_feat = [c for c in df.columns if c.startswith(cfg['prefixo_cols'])]
else:
    cols_feat = [c for c in df.columns if c not in excluir]

X = df[cols_feat].values
y = df['classe'].values

print(f"\nImagens carregadas  : {len(df)}")
print(f"Features por imagem : {len(cols_feat)}  ({cfg['label_cols']})")
print(f"\nDistribuicao:\n{df['classe'].value_counts().to_string()}\n")

mask = {c: y == c for c in ['Saudavel', 'Doente', 'Incerto']}

# 3. SIMILARIDADE COSSENO
"""
Para cada imagem Incerta, calcula a similaridade com o
centroide de Saudavel e o centroide de Doente.

Hipotese: se os incertos tem similaridade ~igual com ambos
os centroides (diferenca ~ 0), a incerteza vem de
semelhanca com as duas classes — nao de ruido ou ambiguidade.
"""
print("Calculando centroides e similaridade cosseno...")

centroides   = {c: X[mask[c]].mean(axis=0) for c in mask}
X_incerto    = X[mask['Incerto']]

sim_saudavel = cosine_similarity(
    X_incerto, centroides['Saudavel'].reshape(1, -1)
).ravel()

sim_doente = cosine_similarity(
    X_incerto, centroides['Doente'].reshape(1, -1)
).ravel()

df_sim = pd.DataFrame({
    'nome_arquivo':  df.loc[mask['Incerto'], 'nome_arquivo'].values,
    'sim_saudavel':  sim_saudavel,
    'sim_doente':    sim_doente,
    'diferenca_abs': np.abs(sim_saudavel - sim_doente),
})
df_sim.to_csv(cfg['output_csv'], index=False)

print(f"\n--- Similaridade cosseno: Incertos vs Centroides ---")
print(f"  vs Saudavel  — media: {sim_saudavel.mean():.6f}  std: {sim_saudavel.std():.6f}")
print(f"  vs Doente    — media: {sim_doente.mean():.6f}  std: {sim_doente.std():.6f}")
print(f"  |diferenca|  — media: {df_sim['diferenca_abs'].mean():.6f}")
print(f"\nArquivo salvo: {cfg['output_csv']}")

# 4. TESTE KRUSKAL-WALLIS
"""
Testa feature a feature se as 3 classes tem distribuicoes
estatisticamente diferentes.
p < 0.05: aquela feature diferencia estatisticamente as classes.
"""
print(f"\n--- Teste Kruskal-Wallis ({cfg['label_cols']}) ---")

ps     = []
labels = []
for col in cols_feat:
    grupos = [df.loc[mask[c], col].values for c in ['Saudavel', 'Doente', 'Incerto']]
    _, p = kruskal(*grupos)
    ps.append(p)
    labels.append(col)

ps  = np.array(ps)
sig = (ps < 0.05).sum()

print(f"  Significativos (p < 0.05) : {sig} / {len(ps)}")
print(f"  p-value minimo            : {ps.min():.2e}")
print(f"  p-value medio             : {ps.mean():.4f}")

if len(cols_feat) <= 10:
    print()
    for lbl, pv in zip(labels, ps):
        sinal = "OK" if pv < 0.05 else "XX"
        print(f"  [{sinal}]  {lbl:<22} p = {pv:.2e}")

# 5. INTERPRETACAO AUTOMATICA
outro     = 'GLCM' if DESCRITOR == 'LBP' else 'LBP'
dif_media = df_sim['diferenca_abs'].mean()

if dif_media < 0.01:
    interpretacao = "CONFIRMA hipotese: incerteza por alta similaridade com ambas as classes."
elif dif_media < 0.02:
    interpretacao = "SUGERE: incerteza possivelmente por similaridade, mas com variacao."
else:
    interpretacao = "NAO confirma hipotese de forma clara — investigar mais."

print("\n" + "=" * 60)
print("  RESUMO")
print("=" * 60)
print(f"\n  Descritor            : {DESCRITOR}")
print(f"  |dif| media          : {dif_media:.6f}")
print(f"  Signif. (KW p<0.05) : {sig}/{len(ps)}")
print(f"  Interpretacao        : {interpretacao}")
print(f"\n  Para rodar com {outro}: altere DESCRITOR = '{outro}' no inicio do script.")
print("=" * 60)
