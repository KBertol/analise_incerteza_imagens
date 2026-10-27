import os
import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

# 1. CONFIGURAÇÕES
INVENTARIO_CSV  = 'inventario_base_dados.csv'
OUTPUT_CSV      = 'features_glcm.csv'

# Mantendo o mesmo pré-processamento do LBP para comparação 
OUTPUT_SIZE = (512, 512)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Configurações GLCM
DISTANCIAS = [5] 
ANGULOS = [0, np.pi/4, np.pi/2, 3*np.pi/4] 

# 2. FUNÇÕES
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    img = clahe.apply(img)
    return img 

def main():
    print("=" * 55)
    print("  EXTRAÇÃO DE FEATURES — GLCM")
    print("=" * 55)

    if not os.path.exists(INVENTARIO_CSV):
        print(f"[ERRO] Arquivo '{INVENTARIO_CSV}' não encontrado.")
        return

    df = pd.read_csv(INVENTARIO_CSV)
    registros = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extraindo GLCM"):
        img = preprocess_image(row['caminho'])
        if img is None: continue
        glcm = graycomatrix(img, distances=DISTANCIAS, angles=ANGULOS, 
                            levels=256, symmetric=True, normed=True)

        registro = {
            'nome_arquivo': row['nome_arquivo'],
            'classe':       row['classe'],
            'contraste':      np.mean(graycoprops(glcm, 'contrast')),
            'dissimilaridade': np.mean(graycoprops(glcm, 'dissimilarity')),
            'homogeneidade':   np.mean(graycoprops(glcm, 'homogeneity')),
            'energia':         np.mean(graycoprops(glcm, 'energy')),
            'correlacao':      np.mean(graycoprops(glcm, 'correlation')),
            'entropia':        -np.sum(glcm * np.log2(glcm + (glcm==0))) # Medida de desordem
        }
        registros.append(registro)

    df_features = pd.DataFrame(registros)
    df_features.to_csv(OUTPUT_CSV, index=False)
    print(f"\nConcluído! Salvo em: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
