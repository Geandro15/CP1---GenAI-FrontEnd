# treinar_modelo.py
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# ─── Configurações ───────────────────────────────────────────────────────
PASTA_NORMAL    = "imagens/normal"
PASTA_PNEUMONIA = "imagens/pneumonia"
PASTA_MODELOS   = "models"
LATENT_DIM      = 16
EPOCHS          = 50
BATCH_SIZE      = 8

# ─── Função para carregar imagens de uma pasta ───────────────────────────
def carregar_imagens(pasta, nome):
    extensoes_validas = {".png", ".jpg", ".jpeg"}
    caminhos = [
        p for p in Path(pasta).iterdir()
        if p.suffix.lower() in extensoes_validas
    ]

    if len(caminhos) == 0:
        print(f"  ERRO: Nenhuma imagem encontrada em '{pasta}'")
        return None

    imagens = []
    erros   = 0

    for caminho in caminhos:
        try:
            img = Image.open(caminho).convert("L")
            img = img.resize((28, 28))
            arr = np.array(img) / 255.0
            imagens.append(arr)
        except Exception as e:
            print(f"  AVISO: erro ao carregar {caminho.name}: {e}")
            erros += 1

    print(f"  {nome}: {len(imagens)} imagens carregadas", end="")
    if erros > 0:
        print(f" ({erros} erros)", end="")
    print()

    return np.array(imagens)[..., np.newaxis].astype(np.float32)

# ─── Função para construir o VAE ─────────────────────────────────────────
def construir_vae(latent_dim):

    def build_encoder():
        inp = tf.keras.Input((28, 28, 1))
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inp)
        x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        z_mean    = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
        z = tf.keras.layers.Lambda(
            lambda t: t[0] + tf.exp(0.5 * t[1]) * tf.random.normal(tf.shape(t[0])),
            name="z"
        )([z_mean, z_log_var])
        return tf.keras.Model(inp, [z, z_mean, z_log_var], name="encoder")

    def build_decoder():
        inp = tf.keras.Input((latent_dim,))
        x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu")(inp)
        x = tf.keras.layers.Reshape((7, 7, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
        out = tf.keras.layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)
        return tf.keras.Model(inp, out, name="decoder")

    class VAE(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.encoder = build_encoder()
            self.decoder = build_decoder()

        def call(self, x):
            z, _, _ = self.encoder(x)
            return self.decoder(z)

        def train_step(self, data):
            x, _ = data
            with tf.GradientTape() as tape:
                z, z_mean, z_log_var = self.encoder(x, training=True)
                reconstruction = self.decoder(z, training=True)

                perda_reconstrucao = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(x, reconstruction)
                ) * 28 * 28

                perda_kl = -0.5 * tf.reduce_mean(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                )

                perda_total = perda_reconstrucao + perda_kl

            gradientes = tape.gradient(perda_total, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradientes, self.trainable_variables))
            return {"perda": perda_total, "reconstrucao": perda_reconstrucao, "kl": perda_kl}

    vae = VAE()
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    vae(tf.zeros((1, 28, 28, 1)))
    return vae

# ─── INÍCIO DO TREINAMENTO ───────────────────────────────────────────────
print("=" * 50)
print("TREINAMENTO DO MODELO VAE — VERSÃO MELHORADA")
print("=" * 50)

# ─── Verifica pastas ─────────────────────────────────────────────────────
for pasta in [PASTA_NORMAL, PASTA_PNEUMONIA]:
    if not os.path.exists(pasta):
        print(f"\nERRO: Pasta '{pasta}' não encontrada!")
        print(f"Crie a pasta '{pasta}' e coloque as imagens dentro.")
        exit(1)

# ─── Carrega imagens ─────────────────────────────────────────────────────
print("\n[1/5] Carregando imagens...")
X_normal    = carregar_imagens(PASTA_NORMAL,    "Normal")
X_pneumonia = carregar_imagens(PASTA_PNEUMONIA, "Pneumonia")

if X_normal is None or X_pneumonia is None:
    print("\nERRO: Verifique se as pastas contêm imagens PNG/JPG.")
    exit(1)

# ─── Aviso de desbalanço ─────────────────────────────────────────────────
total_n = len(X_normal)
total_p = len(X_pneumonia)
razao   = max(total_n, total_p) / max(min(total_n, total_p), 1)

print(f"\n  Total normais:    {total_n}")
print(f"  Total pneumonia:  {total_p}")

if razao > 3:
    print(f"\n  AVISO: Dataset desbalanceado (razão {razao:.1f}x).")
    print("  O data augmentation vai compensar a diferença.")

# ─── Data augmentation nas imagens normais ───────────────────────────────
# Cria variações das imagens para compensar a pouca quantidade
print("\n[2/5] Aplicando data augmentation nas imagens normais...")

def augmentar(imagens, multiplicador=5):
    """
    Cria versões variadas de cada imagem:
    - Espelhamento horizontal aleatório
    - Ruído leve para simular variações de equipamento
    - Variação de brilho
    multiplicador=5 significa que cada imagem vira 5 imagens diferentes.
    """
    aumentadas = []
    for img in imagens:
        for _ in range(multiplicador):
            nova = img.copy()

            # Espelhamento horizontal aleatório
            if np.random.rand() > 0.5:
                nova = nova[:, ::-1, :]

            # Ruído leve
            ruido = np.random.normal(0, 0.02, nova.shape).astype(np.float32)
            nova  = np.clip(nova + ruido, 0, 1)

            # Variação de brilho
            fator = np.random.uniform(0.9, 1.1)
            nova  = np.clip(nova * fator, 0, 1)

            aumentadas.append(nova)

    return np.array(aumentadas, dtype=np.float32)

X_normal_aug = augmentar(X_normal, multiplicador=5)
print(f"  Normais originais:  {len(X_normal)}")
print(f"  Normais após aug:   {len(X_normal_aug)}")

# ─── Treina VAE SOMENTE com imagens normais ───────────────────────────────
# O VAE aprende o padrão de pulmão saudável.
# Imagens com pneumonia serão mal reconstruídas → MSE alto → detectadas.
print(f"\n[3/5] Treinando VAE com imagens NORMAIS ({len(X_normal_aug)} imagens)...")
print("  (O modelo aprende o padrão de pulmão saudável)\n")

vae_normal = construir_vae(LATENT_DIM)
vae_normal.fit(
    X_normal_aug, X_normal_aug,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=1
)

# ─── Calcula limiares recomendados automaticamente ───────────────────────
print("\n[4/5] Calculando limiar recomendado...")

def calcular_mse(modelo, imagens):
    """Calcula o MSE de cada imagem e retorna a lista."""
    mses = []
    for img in imagens:
        x   = img[np.newaxis, ...]
        rec = modelo(x).numpy()
        mse = float(np.mean((x - rec) ** 2))
        mses.append(mse)
    return np.array(mses)

mse_normais    = calcular_mse(vae_normal, X_normal)
mse_pneumonias = calcular_mse(vae_normal, X_pneumonia)

media_normal    = float(np.mean(mse_normais))
media_pneumonia = float(np.mean(mse_pneumonias))
limiar_sugerido = float((media_normal + media_pneumonia) / 2)

print(f"  MSE médio — normais:    {media_normal:.4f}")
print(f"  MSE médio — pneumonia:  {media_pneumonia:.4f}")
print(f"  Limiar sugerido:        {limiar_sugerido:.4f}")

if media_normal >= media_pneumonia:
    print("\n  AVISO: Os MSEs ficaram muito parecidos.")
    print("  O modelo pode não estar distinguindo bem.")
    print("  Tente adicionar mais imagens normais e retreinar.")

# ─── Salva modelo e configurações ────────────────────────────────────────
print(f"\n[5/5] Salvando modelo em '{PASTA_MODELOS}/'...")

os.makedirs(PASTA_MODELOS, exist_ok=True)

caminho_pesos  = os.path.join(PASTA_MODELOS, "vae_pneumonia.weights.h5")
caminho_config = os.path.join(PASTA_MODELOS, "config.json")

vae_normal.save_weights(caminho_pesos)

with open(caminho_config, "w") as f:
    json.dump({
        "latent_dim":       LATENT_DIM,
        "limiar_sugerido":  round(limiar_sugerido, 4),
        "mse_normal_medio": round(media_normal,    4),
        "mse_pneum_medio":  round(media_pneumonia, 4),
    }, f, indent=2)

print(f"  Pesos salvos:  {caminho_pesos}")
print(f"  Config salva:  {caminho_config}")

# ─── Resumo final ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("CONCLUÍDO!")
print("=" * 50)
print(f"  Imagens normais usadas:     {len(X_normal_aug)}")
print(f"  Épocas de treinamento:      {EPOCHS}")
print(f"  MSE médio normais:          {media_normal:.4f}")
print(f"  MSE médio pneumonia:        {media_pneumonia:.4f}")
print(f"  Limiar sugerido para o app: {limiar_sugerido:.4f}")
print(f"\n  No app, ajuste o slider para: {limiar_sugerido:.4f}")
print("\nAgora execute:")
print("  streamlit run front.py")
print("=" * 50)

# Para melhor analise, futuramente utilizaremos um dataset na bibliteca do Kaggle:
# Chest X-Ray Images Pneumonia — Kaggle Ele tem mais de 5000 imagens normais e pneumonia 