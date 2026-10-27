import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay)

# 0. REPRODUTIBILIDADE
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 1. CARREGAR FEATURES
print("=" * 60)
print("  PARTE 2 — ANÁLISE DE INCERTEZA (REGRESSÃO LOGÍSTICA)")
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

print(f"\nFeatures         : {len(cols_feat)}")
print(f"Treino disponível: {len(df_certos)} imagens (Saudavel + Doente)")
print(f"Incertos         : {len(df_incerto)} imagens")

# 2. PRÉ-PROCESSAMENTO
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_certos)
X_inc_sc = scaler.transform(X_incerto)  # mesma escala do treino

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_scaled, y_certos,
    test_size=0.2,
    random_state=SEED,
    stratify=y_certos,
)

# 3. TREINAR MODELO
modelo = LogisticRegression(
    solver='lbfgs',
    C=1.0,
    class_weight='balanced',  # compensar diferença de tamanho entre classes
    max_iter=1000,
    random_state=SEED,
)
modelo.fit(X_treino, y_treino)

# 4. AVALIAÇÃO
y_pred   = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, y_pred)
cv       = cross_val_score(modelo, X_scaled, y_certos, cv=5, scoring='accuracy')

print(f"\n{'─'*60}")
print("  AVALIAÇÃO DO MODELO")
print(f"{'─'*60}")
print(f"  Acurácia no teste (20%) : {acuracia*100:.2f}%")
print(f"  Validação cruzada 5-fold: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
print(f"\n{classification_report(y_teste, y_pred, digits=3)}")

# 5. ANÁLISE DOS INCERTOS
proba   = modelo.predict_proba(X_inc_sc)
classes = list(modelo.classes_)
idx_d   = classes.index('Doente')
idx_s   = classes.index('Saudavel')

df_incerto['prob_doente']   = proba[:, idx_d]
df_incerto['prob_saudavel'] = proba[:, idx_s]
df_incerto['pred_classe']   = modelo.predict(X_inc_sc)
df_incerto['margem']        = np.abs(proba[:, idx_d] - proba[:, idx_s])

"""
logit(p) = log(p / 1-p)
logit ≈ 0 -> p ≈ 0.5 -> máxima incerteza
logit > 0 -> modelo pende para Doente
logit < 0 -> modelo pende para Saudavel
"""
df_incerto['logit_doente'] = np.log(
    df_incerto['prob_doente'] / (1 - df_incerto['prob_doente'] + 1e-10)
)

df_incerto[['nome_arquivo', 'prob_saudavel', 'prob_doente',
            'pred_classe', 'margem', 'logit_doente']]\
    .to_csv('incertos_softmax.csv', index=False)

contagem = df_incerto['pred_classe'].value_counts()
total    = len(df_incerto)
limbo    = (df_incerto['margem'] <= 0.20).sum() 

print(f"{'─'*60}")
print("  CLASSIFICAÇÃO DOS INCERTOS")
print(f"{'─'*60}")
print(f"  Classificados como Doente   : {contagem.get('Doente',0)}  ({contagem.get('Doente',0)/total*100:.1f}%)")
print(f"  Classificados como Saudavel : {contagem.get('Saudavel',0)}  ({contagem.get('Saudavel',0)/total*100:.1f}%)")
print(f"  No limbo (40%–60%)          : {limbo}  ({limbo/total*100:.1f}%)")
print(f"  Logit médio                 : {df_incerto['logit_doente'].mean():.4f}")
print(f"\nArquivo salvo: incertos_softmax.csv")

# 6. GRÁFICOS
# Gráfico 1: Matriz de confusão
cm = confusion_matrix(y_teste, y_pred, labels=modelo.classes_)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)
disp.plot(cmap='Greys', values_format='d', ax=ax)
ax.set_title(f'Matriz de Confusão — Regressão Logística\nAcurácia: {acuracia*100:.1f}%', fontsize=13)
plt.tight_layout()
plt.savefig('logit_matriz_confusao.png', dpi=150)
plt.close()
print("\nSalvo: logit_matriz_confusao.png")

# Gráfico 2: Distribuição softmax dos incertos
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df_incerto['prob_doente'], bins=40, kde=True,
             color='#333333', ax=ax, label='P(Doente)')
ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5,
           label='Incerteza máxima (50/50)')
ax.set_title('Distribuição de Probabilidades Softmax — Imagens Incertas', fontsize=13)
ax.set_xlabel('P(Doente)  —  valores próximos a 0.5 indicam alta incerteza')
ax.set_ylabel('Número de imagens')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('softmax_distribuicao.png', dpi=150)
plt.close()
print("Salvo: softmax_distribuicao.png")

# Gráfico 3: Distribuição do logit
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df_incerto['logit_doente'], bins=40, kde=True,
             color='#555555', ax=ax)
ax.axvline(0, color='black', linestyle='--', linewidth=1.5,
           label='logit = 0  (máxima incerteza)')
ax.axvline(df_incerto['logit_doente'].mean(), color='gray', linestyle='-',
           linewidth=1.2, label=f"média = {df_incerto['logit_doente'].mean():.3f}")
ax.set_title('Distribuição do Logit — Imagens Incertas\nlogit ≈ 0 → máxima incerteza  |  logit > 0 → pende para Doente', fontsize=13)
ax.set_xlabel('logit(P(Doente))')
ax.set_ylabel('Número de imagens')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('logit_distribuicao.png', dpi=150)
plt.close()
print("Salvo: logit_distribuicao.png")

# 7. RESUMO FINAL
print(f"\n{'='*60}")
print("  RESUMO FINAL")
print(f"{'='*60}")
print(f"  Acurácia teste          : {acuracia*100:.2f}%")
print(f"  CV 5-fold               : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
print(f"  Incertos → Doente       : {contagem.get('Doente',0)} ({contagem.get('Doente',0)/total*100:.1f}%)")
print(f"  Incertos → Saudavel     : {contagem.get('Saudavel',0)} ({contagem.get('Saudavel',0)/total*100:.1f}%)")
print(f"  No limbo (40%–60%)      : {limbo} ({limbo/total*100:.1f}%)")
print(f"  Logit médio             : {df_incerto['logit_doente'].mean():.4f}")
print(f"{'='*60}")
print(f"\n  Interpretação do logit:")
print(f"  logit ≈ 0 → p ≈ 0.5 → máxima incerteza")
print(f"  logit > 0 → modelo pende para Doente")
print(f"  logit < 0 → modelo pende para Saudavel")
