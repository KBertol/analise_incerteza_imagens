import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 0. REPRODUTIBILIDADE
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 1. CARREGAR E UNIR FEATURES
print("=" * 60)
print("  RANDOM FOREST — LBP + GLCM")
print("=" * 60)

df_lbp  = pd.read_csv('features_lbp.csv')
df_glcm = pd.read_csv('features_glcm.csv')
df      = pd.merge(df_lbp, df_glcm, on=['nome_arquivo', 'classe'])

excluir   = ['nome_arquivo', 'classe', 'caminho']
cols_feat = [c for c in df.columns if c not in excluir]

n_lbp  = len([c for c in cols_feat if c.startswith('lbp')])
n_glcm = len([c for c in cols_feat if not c.startswith('lbp')])

print(f"\nImagens no dataset     : {len(df)}")
print(f"Features combinadas    : {len(cols_feat)}  (LBP: {n_lbp}  |  GLCM: {n_glcm})")

# 2. SEPARAR CONJUNTOS
df_certos  = df[df['classe'].isin(['Saudavel', 'Doente'])].copy()
df_incerto = df[df['classe'] == 'Incerto'].copy()

X_certos = df_certos[cols_feat].values
y_certos = df_certos['classe'].values

X_incerto = df_incerto[cols_feat].values

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_certos, y_certos,
    test_size=0.2,
    random_state=SEED,
    stratify=y_certos,   
)

print(f"\nTreino : {len(X_treino)} imagens  |  Teste: {len(X_teste)} imagens")
print(f"Incerto: {len(X_incerto)} imagens (não vistos durante treino)")

# 3. TREINAR MODELO
modelo = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced', 
    random_state=SEED,
    n_jobs=-1,
)
modelo.fit(X_treino, y_treino)

# 4. AVALIAÇÃO NO CONJUNTO DE TESTE
y_pred    = modelo.predict(X_teste)
acuracia  = accuracy_score(y_teste, y_pred)
cv_scores = cross_val_score(modelo, X_certos, y_certos,
                             cv=5, scoring='accuracy', n_jobs=-1)

print(f"\n{'─'*60}")
print("  AVALIAÇÃO DO MODELO")
print(f"{'─'*60}")
print(f"  Acurácia no teste (20%) : {acuracia*100:.2f}%")
print(f"  Validação cruzada 5-fold: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"\n  Relatório de Classificação:")
print(classification_report(y_teste, y_pred, digits=3))

# 5. CLASSIFICAR OS INCERTOS
proba        = modelo.predict_proba(X_incerto)
classes_mod  = list(modelo.classes_)
idx_d        = classes_mod.index('Doente')
idx_s        = classes_mod.index('Saudavel')

df_incerto['prob_doente']   = proba[:, idx_d]
df_incerto['prob_saudavel'] = proba[:, idx_s]
df_incerto['pred_classe']   = modelo.predict(X_incerto)
df_incerto['incerteza']     = np.abs(proba[:, idx_d] - proba[:, idx_s])

df_incerto[['nome_arquivo', 'prob_saudavel', 'prob_doente',
            'pred_classe', 'incerteza']].to_csv('incertos_probabilidades.csv', index=False)

contagem   = df_incerto['pred_classe'].value_counts()
total      = len(df_incerto)
perc_d     = contagem.get('Doente', 0) / total * 100
perc_s     = contagem.get('Saudavel', 0) / total * 100
inc_media  = df_incerto['incerteza'].mean()
quase_5050 = (df_incerto['incerteza'] < 0.2).sum()

print(f"{'─'*60}")
print("  CLASSIFICAÇÃO DOS INCERTOS")
print(f"{'─'*60}")
print(f"  Classificados como Doente  : {contagem.get('Doente',0)}  ({perc_d:.1f}%)")
print(f"  Classificados como Saudavel: {contagem.get('Saudavel',0)}  ({perc_s:.1f}%)")
print(f"  Incerteza média            : {inc_media:.4f}")
print(f"  Com incerteza < 0.2 (≈50/50): {quase_5050}  ({quase_5050/total*100:.1f}%)")

if perc_d > 60:
    conclusao = "O modelo pende para DOENTE, mas com baixa confiança."
elif perc_s > 60:
    conclusao = "O modelo pende para SAUDÁVEL, mas com baixa confiança."
else:
    conclusao = "O modelo ficou dividido (próximo a 50/50)."
print(f"\n  {conclusao}")

# 6. IMPORTÂNCIA DAS FEATURES
df_imp = pd.DataFrame({
    'Feature':     cols_feat,
    'Importancia': modelo.feature_importances_,
}).sort_values('Importancia', ascending=False).reset_index(drop=True)

print(f"\n{'─'*60}")
print("  TOP 10 FEATURES MAIS IMPORTANTES")
print(f"{'─'*60}")
print(df_imp.head(10).to_string(index=False))

# 7. GRÁFICOS
# Gráfico 1: Importância das features 
fig, ax = plt.subplots(figsize=(12, 8))
top15 = df_imp.head(15)
cores = ['#2196F3' if 'lbp' in f else '#F44336' for f in top15['Feature']]

ax.barh(top15['Feature'][::-1], top15['Importancia'][::-1],
        color=cores[::-1], edgecolor='white', linewidth=0.5)

legenda = [Patch(color='#2196F3', label='LBP'), Patch(color='#F44336', label='GLCM')]
ax.legend(handles=legenda, fontsize=11)
ax.set_title('Top 15 Features — Random Forest (LBP + GLCM)\nImportância Gini', fontsize=13)
ax.set_xlabel('Importância', fontsize=11)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('importancia_features.png', dpi=150)
plt.close()
print("\nSalvo: importancia_features.png")

# Gráfico 2: Matriz de confusão 
cm = confusion_matrix(y_teste, y_pred, labels=['Saudavel', 'Doente'])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)

classes_labels = ['Saudavel', 'Doente']
ticks = range(len(classes_labels))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(classes_labels, fontsize=11)
ax.set_yticklabels(classes_labels, fontsize=11)

thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center', fontsize=13,
                color='white' if cm[i, j] > thresh else 'black')

ax.set_title(f'Matriz de Confusão — conjunto teste (20%)\nAcurácia: {acuracia*100:.2f}%', fontsize=12)
ax.set_xlabel('Predito', fontsize=11)
ax.set_ylabel('Real', fontsize=11)
plt.tight_layout()
plt.savefig('matriz_confusao_rand_forest.png', dpi=150)
plt.close()
print("Salvo: matriz_confusao_rand_forest.png")

# Gráfico 3: Distribuição de probabilidades dos incertos
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df_incerto['prob_saudavel'], df_incerto['prob_doente'],
           c=df_incerto['incerteza'], cmap='YlOrRd', alpha=0.5, s=14)
plt.colorbar(
    plt.cm.ScalarMappable(cmap='YlOrRd',
                          norm=plt.Normalize(df_incerto['incerteza'].min(),
                                             df_incerto['incerteza'].max())),
    ax=ax, label='|P_Doente - P_Saudavel|'
)
lim = [0, 1]
ax.plot(lim, [1 - x for x in lim], 'k--', alpha=0.3, linewidth=1,
        label='linha P_D + P_S = 1')
ax.set_title('Incertos — Probabilidades Random Forest\nP(Saudavel) x P(Doente)', fontsize=12)
ax.set_xlabel('P(Saudavel)', fontsize=11)
ax.set_ylabel('P(Doente)', fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('incertos_probabilidades.png', dpi=150)
plt.close()
print("Salvo: incertos_probabilidades.png")

# Gráfico 4: Histograma de incerteza
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df_incerto['incerteza'], bins=30, color='#FF9800',
        edgecolor='white', linewidth=0.5)
ax.axvline(0.2, color='#333333', linestyle='--', linewidth=1.5,
           label=f'threshold 0.2  ({quase_5050/total*100:.1f}% abaixo)')
ax.axvline(inc_media, color='#F44336', linestyle='-', linewidth=1.5,
           label=f'media = {inc_media:.4f}')
ax.set_title('Incertos — Distribuicao de Incerteza\n|P(Doente) - P(Saudavel)|', fontsize=12)
ax.set_xlabel('Incerteza  |P_D - P_S|', fontsize=11)
ax.set_ylabel('Numero de imagens', fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('incertos_incerteza.png', dpi=150)
plt.close()
print("Salvo: incertos_incerteza.png")

# 8. RESUMO FINAL
print(f"\n{'='*60}")
print("  RESUMO FINAL")
print(f"{'='*60}")
print(f"  Acurácia teste          : {acuracia*100:.2f}%")
print(f"  CV 5-fold               : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"  Feature mais importante : {df_imp.iloc[0]['Feature']} ({df_imp.iloc[0]['Importancia']:.4f})")
print(f"  Incertos → Doente       : {contagem.get('Doente',0)} ({perc_d:.1f}%)")
print(f"  Incertos → Saudavel     : {contagem.get('Saudavel',0)} ({perc_s:.1f}%)")
print(f"  Incerteza média         : {inc_media:.4f}")
print(f"  Incertos quase 50/50    : {quase_5050} ({quase_5050/total*100:.1f}%)")
print(f"\n  Arquivos gerados:")
print(f"    - importancia_features.png")
print(f"    - matriz_confusao.png")
print(f"    - incertos_probabilidades.png")
print(f"    - incertos_incerteza.png")
print(f"    - incertos_probabilidades.csv")
print(f"{'='*60}")
