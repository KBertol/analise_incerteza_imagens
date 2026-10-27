import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

def preprocess_image(image_path, output_size=(512, 512)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # Redimensionar
    img_res = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)
    
    # CLAHE 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_res)
    
    # Normalização
    img_norm = img_clahe.astype(np.float32) / 255.0
    return img_res, img_clahe

df = pd.read_csv('inventario_base_dados.csv')

amostras = df.groupby('classe').head(1)

# Plotar os resultados
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plt.subplots_adjust(hspace=0.4)
fig.suptitle('Efeito do CLAHE por classe\n(pré-processamento aplicado em LBP e GLCM)',
             fontsize=13, y=0.98)


for i, (index, row) in enumerate(amostras.iterrows()):
    original, tratada = preprocess_image(row['caminho'])
    
    # Coluna 1: Original (apenas redimensionada)
    axes[i, 0].imshow(original, cmap='gray')
    axes[i, 0].set_title(f"Original: {row['classe']}")
    axes[i, 0].axis('off')
    
    # Coluna 2: Após CLAHE
    axes[i, 1].imshow(tratada, cmap='gray')
    axes[i, 1].set_title(f"Com CLAHE: {row['classe']}")
    axes[i, 1].axis('off')

print("Exibindo imagens... Feche a janela da imagem para encerrar.")
plt.show()
