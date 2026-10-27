import os
import random
import numpy as np
import pandas as pd
import cv2
from skimage.feature import local_binary_pattern
from tqdm import tqdm

# 0. REPRODUTIBILIDADE
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 1. CONFIGURAÇÕES
INVENTARIO_CSV  = 'inventario_base_dados.csv'
OUTPUT_CSV      = 'features_lbp.csv'

OUTPUT_SIZE = (512, 512)
CLAHE_CLIP_LIMIT   = 2.0
CLAHE_TILE_GRID    = (8, 8)

LBP_RADIUS  = 3      
LBP_POINTS  = 8 * LBP_RADIUS 
LBP_METHOD  = 'uniform'

# 2. FUNÇÕES
def preprocess_image(image_path, output_size=OUTPUT_SIZE):
    """
    Lê, redimensiona, aplica CLAHE e normaliza a imagem.
    Retorna array normalizado [0,1] ou None se falhar.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    # Redimensionar
    img = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)

    # CLAHE para melhorar contraste local antes do LBP 
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    img = clahe.apply(img)

    # Normalizar para [0, 1]
    img = img.astype(np.float32) / 255.0

    return img


def extrair_lbp(img_norm):
    """
    Extrai o histograma LBP de uma imagem normalizada [0,1].

    Parâmetros utilizados:
      - radius  = LBP_RADIUS  (vizinhança)
      - npoints = LBP_POINTS  (amostras na circunferência)
      - method  = 'uniform'   (padrões uniformes — reduz de 2^P para P*(P-1)+3 bins)

    Retorna vetor 1D normalizado (histograma de probabilidade).
    """
    img_uint8 = (img_norm * 255).astype(np.uint8)

    lbp = local_binary_pattern(img_uint8, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)

    n_bins = LBP_POINTS + 2

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return hist  

# 3. PIPELINE PRINCIPAL
def main():
    print("=" * 55)
    print("  EXTRAÇÃO DE FEATURES — LBP")
    print(f"  Seed: {SEED} | Radius: {LBP_RADIUS} | Points: {LBP_POINTS}")
    print(f"  Method: {LBP_METHOD} | Bins: {LBP_POINTS + 2}")
    print("=" * 55)

    if not os.path.exists(INVENTARIO_CSV):
        print(f"\n[ERRO] Arquivo '{INVENTARIO_CSV}' não encontrado.")
        print("Execute o script padrao.py primeiro para gerar o inventário.")
        return

    df = pd.read_csv(INVENTARIO_CSV)
    print(f"\nImagens no inventário: {len(df)}")
    print(df['classe'].value_counts().to_string())
    print()

    registros = []
    erros     = []

    n_bins = LBP_POINTS + 2
    colunas_lbp = [f'lbp_bin_{i:02d}' for i in range(n_bins)]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extraindo LBP"):
        caminho = row['caminho']
        classe  = row['classe']
        nome    = row['nome_arquivo']

        # Pré-processar
        img = preprocess_image(caminho)

        if img is None:
            erros.append(caminho)
            continue

        # Extrair LBP
        hist_lbp = extrair_lbp(img)

        # Montar registro
        registro = {
            'nome_arquivo': nome,
            'caminho':      caminho,
            'classe':       classe,
        }
        for i, val in enumerate(hist_lbp):
            registro[colunas_lbp[i]] = val

        registros.append(registro)

    # Salvar CSV
    df_features = pd.DataFrame(registros)
    df_features.to_csv(OUTPUT_CSV, index=False)

    # Relatório final
    print("\n" + "=" * 55)
    print(f"  Concluído! Features salvas em: {OUTPUT_CSV}")
    print(f"  Imagens processadas: {len(registros)}")
    print(f"  Imagens com erro:    {len(erros)}")
    print(f"  Shape do CSV:        {df_features.shape}")
    print("=" * 55)

    if erros:
        print(f"\n[AVISO] {len(erros)} imagens não puderam ser lidas:")
        for e in erros[:10]:
            print(f"  - {e}")
        if len(erros) > 10:
            print(f"  ... e mais {len(erros) - 10}.")

    print("\nDistribuição no CSV final:")
    print(df_features['classe'].value_counts().to_string())


if __name__ == '__main__':
    main()
