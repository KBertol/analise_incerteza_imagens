import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 0. REPRODUTIBILIDADE
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 1. CONFIGURAÇÕES — altere aqui
# Opções: 'LBP' ou 'GLCM'
DESCRITOR = 'LBP'

# Arquivos e parâmetros por descritor
CONFIGS = {
    'LBP': {
        'features_csv':     'features_lbp.csv',
        'similaridade_csv': 'similaridade_incertos.csv',
        'prefixo_cols':     'lbp_bin_',
        'prefixo_saida':    'lbp',
        'titulo_tsne':      't-SNE — Features LBP (raio=3, uniform)',
        'titulo_hist':      'Histograma LBP médio por classe',
        'xlabel_hist':      'Bin LBP',
    },
    'GLCM': {
        'features_csv':     'features_glcm.csv',
        'similaridade_csv': 'similaridade_incertos_glcm.csv',
        'prefixo_cols':     None,  
        'prefixo_saida':    'glcm',
        'titulo_tsne':      't-SNE — Features GLCM (dist=5, 4 angulos)',
        'titulo_hist':      'Features GLCM medias por classe',
        'xlabel_hist':      'Feature GLCM',
    },
}

cfg = CONFIGS[DESCRITOR]

CORES = {
    'Saudavel': '#2196F3',
    'Doente':   '#F44336',
    'Incerto':  '#FF9800',
}

# 2. CARREGAR DADOS
print("=" * 60)
print(f"  VISUALIZACAO — {DESCRITOR}")
print("=" * 60)

df     = pd.read_csv(cfg['features_csv'])
df_sim = pd.read_csv(cfg['similaridade_csv'])

excluir = ['nome_arquivo', 'classe', 'caminho']
if cfg['prefixo_cols']:
    cols_feat = [c for c in df.columns if c.startswith(cfg['prefixo_cols'])]
else:
    cols_feat = [c for c in df.columns if c not in excluir]

X = df[cols_feat].values
y = df['classe'].values
mask = {c: y == c for c in ['Saudavel', 'Doente', 'Incerto']}

print(f"\nImagens carregadas : {len(df)}")
print(f"Features           : {len(cols_feat)}  -> {cols_feat}")
print(f"\nDistribuicao:\n{df['classe'].value_counts().to_string()}\n")

# 3. t-SNE
print("Criando projecao t-SNE (pode levar 1-2 min)...")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(
    n_components=2,
    perplexity=40,
    max_iter=1000,
    random_state=SEED,
    init='pca',
    learning_rate='auto',
)
X_2d = tsne.fit_transform(X_scaled)

# Gráfico 1: t-SNE por classe
fig, ax = plt.subplots(figsize=(10, 8))

for classe, cor in CORES.items():
    idx = mask[classe]
    ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
               c=cor, label=classe, alpha=0.45, s=12, linewidths=0)
    cx, cy = X_2d[idx, 0].mean(), X_2d[idx, 1].mean()
    ax.scatter(cx, cy, c=cor, s=180, marker='*',
               edgecolors='black', linewidths=0.8, zorder=5)

legendas = [mpatches.Patch(color=c, label=l) for l, c in CORES.items()]
ax.legend(handles=legendas, fontsize=11, framealpha=0.9)
ax.set_title(f"{cfg['titulo_tsne']}\nEstrelas = centroide da classe", fontsize=13)
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
ax.grid(True, alpha=0.2)
plt.tight_layout()

saida_tsne = f"tsne_{cfg['prefixo_saida']}.png"
plt.savefig(saida_tsne, dpi=150)
plt.close()
print(f"Salvo: {saida_tsne}")

# Gráfico 2: média das features por classe
fig, ax = plt.subplots(figsize=(10, 5))

for classe, cor in CORES.items():
    media = X[mask[classe]].mean(axis=0)
    ax.plot(range(len(cols_feat)), media,
            color=cor, label=classe, linewidth=2, marker='o', markersize=4)

ax.set_xticks(range(len(cols_feat)))
ax.set_xticklabels(cols_feat, rotation=30, ha='right', fontsize=9)
ax.set_title(cfg['titulo_hist'], fontsize=13)
ax.set_xlabel(cfg['xlabel_hist'])
ax.set_ylabel('Valor medio normalizado')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

saida_hist = f"histograma_{cfg['prefixo_saida']}_classes.png"
plt.savefig(saida_hist, dpi=150)
plt.close()
print(f"Salvo: {saida_hist}")

# Gráfico 3: scatter similaridade dos incertos
fig, ax = plt.subplots(figsize=(8, 8))

sc = ax.scatter(
    df_sim['sim_saudavel'], df_sim['sim_doente'],
    c=df_sim['diferenca_abs'], cmap='YlOrRd', alpha=0.6, s=18,
)
plt.colorbar(sc, ax=ax, label='|sim_saudavel - sim_doente|')

lim = [df_sim[['sim_saudavel', 'sim_doente']].min().min() - 0.001,
       df_sim[['sim_saudavel', 'sim_doente']].max().max() + 0.001]
ax.plot(lim, lim, 'k--', alpha=0.3, linewidth=1, label='igual similaridade')

ax.set_title(
    f'Incertos ({DESCRITOR}): similaridade cosseno\nvs centroide Saudavel x centroide Doente',
    fontsize=13)
ax.set_xlabel('Similaridade com Saudavel')
ax.set_ylabel('Similaridade com Doente')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()

saida_scatter = f"similaridade_scatter_{cfg['prefixo_saida']}.png"
plt.savefig(saida_scatter, dpi=150)
plt.close()
print(f"Salvo: {saida_scatter}")

# 4. RESUMO
outro = 'GLCM' if DESCRITOR == 'LBP' else 'LBP'
print("\n" + "=" * 60)
print(f"  Visualizacoes {DESCRITOR} concluidas.")
print(f"  Arquivos gerados:")
print(f"    - {saida_tsne}")
print(f"    - {saida_hist}")
print(f"    - {saida_scatter}")
print(f"\n  Para gerar os graficos do {outro},")
print(f"  altere DESCRITOR = '{outro}' no inicio do script.")
print("=" * 60)
