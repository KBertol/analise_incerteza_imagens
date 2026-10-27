import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# 0. REPRODUTIBILIDADE
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 1. CARREGAR E PREPARAR DADOS
print("=" * 60)
print("  ANÁLISE DE LIMIAR DE DECISÃO")
print("=" * 60)

df_lbp  = pd.read_csv('features_lbp.csv')
df_glcm = pd.read_csv('features_glcm.csv')
df      = df_lbp.merge(df_glcm, on=['nome_arquivo', 'classe'])

excluir   = ['nome_arquivo', 'classe', 'caminho']
cols_feat = [c for c in df.columns if c not in excluir]

df_certos  = df[df['classe'].isin(['Saudavel', 'Doente'])].copy()
df_incerto = df[df['classe'] == 'Incerto'].copy()

X_certos  = df_certos[cols_feat].values
y_certos  = df_certos['classe'].values
X_incerto = df_incerto[cols_feat].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_certos)
X_inc_sc = scaler.transform(X_incerto)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_scaled, y_certos, test_size=0.2, random_state=SEED, stratify=y_certos
)

# 2. TREINAR MODELO
modelo = LogisticRegression(
    solver='lbfgs', C=1.0,
    class_weight='balanced',
    max_iter=1000, random_state=SEED,
)
modelo.fit(X_treino, y_treino)

# Probabilidades base
proba_teste   = modelo.predict_proba(X_teste)
proba_incerto = modelo.predict_proba(X_inc_sc)

classes = list(modelo.classes_)
idx_d   = classes.index('Doente')

# P(Doente) para cada imagem
p_doente_teste   = proba_teste[:, idx_d]
p_doente_incerto = proba_incerto[:, idx_d]

# 3. VARRER LIMIARES
""" 
Para cada limiar t:
   - se P(Doente) > t → classifica como Doente
   - se P(Doente) < (1-t) → classifica como Saudavel
   - caso contrário → fica no limbo (ainda ambíguo)
"""
limiares = np.arange(0.1, 0.91, 0.05)

resultados = []

for t in limiares:
    # Aplicar limiar no conjunto de teste (classes conhecidas)
    pred_teste = np.where(p_doente_teste > t, 'Doente', 'Saudavel')
    acc  = accuracy_score(y_teste, pred_teste)
    f1_d = f1_score(y_teste, pred_teste, pos_label='Doente', zero_division=0)
    f1_s = f1_score(y_teste, pred_teste, pos_label='Saudavel', zero_division=0)

    # Aplicar nos incertos
    doente   = (p_doente_incerto > t).sum()
    saudavel = (p_doente_incerto <= t).sum()

    if t > 0.5:
        limbo = ((p_doente_incerto > (1 - t)) & (p_doente_incerto <= t)).sum()
        doente_adj   = (p_doente_incerto > t).sum()
        saudavel_adj = (p_doente_incerto <= (1 - t)).sum()
    else:
        limbo        = 0
        doente_adj   = doente
        saudavel_adj = saudavel

    resultados.append({
        'limiar':        round(t, 2),
        'acuracia':      acc,
        'f1_doente':     f1_d,
        'f1_saudavel':   f1_s,
        'inc_doente':    doente,
        'inc_saudavel':  saudavel,
        'inc_doente_adj':   doente_adj,
        'inc_saudavel_adj': saudavel_adj,
        'limbo':         limbo,
        'perc_doente':   doente / len(df_incerto) * 100,
        'perc_saudavel': saudavel / len(df_incerto) * 100,
        'perc_limbo':    limbo / len(df_incerto) * 100,
    })

df_res = pd.DataFrame(resultados)
df_res.to_csv('analise_limiar.csv', index=False)

# 4. IMPRIMIR TABELA
print(f"\n{'─'*75}")
print(f"  {'Limiar':>7}  {'Acurácia':>9}  {'F1-D':>6}  {'F1-S':>6}  "
      f"{'→Doente':>8}  {'→Saudavel':>10}  {'Limbo':>6}")
print(f"{'─'*75}")

for _, r in df_res.iterrows():
    marker = " ← padrão" if r['limiar'] == 0.50 else ""
    print(f"  {r['limiar']:>7.2f}  {r['acuracia']*100:>8.1f}%  "
          f"{r['f1_doente']:>6.3f}  {r['f1_saudavel']:>6.3f}  "
          f"{r['inc_doente']:>6.0f} ({r['perc_doente']:>4.1f}%)  "
          f"{r['inc_saudavel']:>7.0f} ({r['perc_saudavel']:>4.1f}%)  "
          f"{r['limbo']:>4.0f} ({r['perc_limbo']:>4.1f}%){marker}")

print(f"{'─'*75}")
print(f"\nArquivo salvo: analise_limiar.csv")

# 5. GRÁFICOS
# Gráfico 1: Distribuição de incertos por limiar
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(df_res['limiar'], df_res['perc_doente'],
        'k-o', markersize=5, linewidth=2, label='→ Doente (%)')
ax.plot(df_res['limiar'], df_res['perc_saudavel'],
        'k--s', markersize=5, linewidth=2, label='→ Saudável (%)')
ax.axvline(0.5, color='gray', linestyle=':', linewidth=1.2, label='limiar padrão (0.5)')
ax.set_title('Classificação dos Incertos por Limiar de Decisão\n'
             'Como o corte de probabilidade distribui as imagens incertas', fontsize=13)
ax.set_xlabel('Limiar de decisão  (P(Doente) > limiar → Doente)')
ax.set_ylabel('% dos incertos')
ax.legend(fontsize=11)
ax.set_xticks(np.arange(0.1, 0.95, 0.1))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('limiar_distribuicao_incertos.png', dpi=150)
plt.close()
print("\nSalvo: limiar_distribuicao_incertos.png")

# Gráfico 2: Acurácia nas classes conhecidas por limiar 
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(df_res['limiar'], df_res['acuracia'] * 100,
        'k-o', markersize=5, linewidth=2, label='Acurácia (%)')
ax.plot(df_res['limiar'], df_res['f1_doente'],
        'k--s', markersize=4, linewidth=1.5, label='F1 Doente')
ax.plot(df_res['limiar'], df_res['f1_saudavel'],
        'k:^', markersize=4, linewidth=1.5, label='F1 Saudável')
ax.axvline(0.5, color='gray', linestyle=':', linewidth=1.2, label='limiar padrão (0.5)')
ax.set_title('Desempenho do Modelo por Limiar de Decisão\n'
             '(avaliado nas classes conhecidas — Saudável e Doente)', fontsize=13)
ax.set_xlabel('Limiar de decisão')
ax.set_ylabel('Métrica')
ax.legend(fontsize=11)
ax.set_xticks(np.arange(0.1, 0.95, 0.1))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('limiar_desempenho.png', dpi=150)
plt.close()
print("Salvo: limiar_desempenho.png")

# Gráfico 3: Histograma de P(Doente) dos incertos
fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(p_doente_incerto, bins=50, color='#444444',
        edgecolor='white', linewidth=0.4)
ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5,
           label='limiar 0.5 (padrão)')
for t_ref in [0.3, 0.7]:
    ax.axvline(t_ref, color='gray', linestyle=':', linewidth=1.2,
               label=f'limiar {t_ref}')
ax.set_title('Distribuição de P(Doente) para os Incertos\n'
             'Linhas verticais mostram onde diferentes limiares cortam', fontsize=13)
ax.set_xlabel('P(Doente)')
ax.set_ylabel('Número de imagens')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('limiar_histograma_incertos.png', dpi=150)
plt.close()
print("Salvo: limiar_histograma_incertos.png")

# 6. RESUMO
melhor = df_res.loc[df_res['acuracia'].idxmax()]
print(f"\n{'='*60}")
print("  RESUMO")
print(f"{'='*60}")
print(f"  Limiar padrão (0.5):")
r05 = df_res[df_res['limiar'] == 0.50].iloc[0]
print(f"    Acurácia  : {r05['acuracia']*100:.1f}%")
print(f"    → Doente  : {r05['perc_doente']:.1f}%")
print(f"    → Saudável: {r05['perc_saudavel']:.1f}%")
print(f"\n  Melhor acurácia encontrada: limiar = {melhor['limiar']:.2f}")
print(f"    Acurácia  : {melhor['acuracia']*100:.1f}%")
print(f"    → Doente  : {melhor['perc_doente']:.1f}%")
print(f"    → Saudável: {melhor['perc_saudavel']:.1f}%")
print(f"\n  Conclusão:")
variacao = df_res['perc_doente'].max() - df_res['perc_doente'].min()
print(f"  Variação de incertos → Doente entre todos os limiares: {variacao:.1f}%")
if variacao < 20:
    print("  A distribuição dos incertos é pouco sensível ao limiar —")
    print("  a ambiguidade persiste independentemente do corte escolhido.")
else:
    print("  O limiar afeta significativamente a distribuição dos incertos.")
print(f"{'='*60}")
