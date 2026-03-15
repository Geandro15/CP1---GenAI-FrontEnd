# Geandro Dezordi    RM 562316
# Alexandre Ferreira RM 565626
# Lucas Veronezi     RM 564202

import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import tensorflow as tf
import json
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─── Configuração da página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Triagem VAE - Raio-X",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Constantes ──────────────────────────────────────────────────────────
CAMINHO_CONFIG   = "models/config.json"
CAMINHO_PESOS    = "models/vae_pneumonia.weights.h5"
FORMATOS_ACEITOS = ["png", "jpg", "jpeg"]

# ─── Inicializa histórico na sessão ──────────────────────────────────────
if "historico" not in st.session_state:
    st.session_state.historico = []

if "paciente_atual" not in st.session_state:
    st.session_state.paciente_atual = ""

# ─── Verifica arquivos do modelo ─────────────────────────────────────────
arquivos_faltando = []
if not os.path.exists(CAMINHO_CONFIG):
    arquivos_faltando.append(CAMINHO_CONFIG)
if not os.path.exists(CAMINHO_PESOS):
    arquivos_faltando.append(CAMINHO_PESOS)

if arquivos_faltando:
    st.error("Arquivos do modelo não encontrados:")
    for f in arquivos_faltando:
        st.code(f)
    st.info("Execute primeiro: python treinar_modelo.py")
    st.stop()

# ─── Carrega modelo ───────────────────────────────────────────────────────
@st.cache_resource
def load_vae_model():
    try:
        with open(CAMINHO_CONFIG, "r") as f:
            latent_dim = json.load(f).get("latent_dim", 16)

        def build_encoder():
            inp = tf.keras.Input((28, 28, 1))
            x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inp)
            x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(128, activation="relu")(x)
            z_mean    = tf.keras.layers.Dense(latent_dim)(x)
            z_log_var = tf.keras.layers.Dense(latent_dim)(x)
            z = tf.keras.layers.Lambda(
                lambda t: t[0] + tf.exp(0.5 * t[1]) * tf.random.normal(tf.shape(t[0]))
            )([z_mean, z_log_var])
            return tf.keras.Model(inp, z, name="encoder")

        def build_decoder():
            inp = tf.keras.Input((latent_dim,))
            x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu")(inp)
            x = tf.keras.layers.Reshape((7, 7, 64))(x)
            x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
            x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
            out = tf.keras.layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)
            return tf.keras.Model(inp, out, name="decoder")

        encoder = build_encoder()
        decoder = build_decoder()

        class MinimalVAE(tf.keras.Model):
            def call(self, x):
                return decoder(encoder(x))

        vae = MinimalVAE()
        vae(tf.zeros((1, 28, 28, 1)))
        vae.load_weights(CAMINHO_PESOS)
        return vae, None
    except Exception as e:
        return None, str(e)

vae, erro_load = load_vae_model()

if erro_load:
    st.error("Falha ao carregar o modelo.")
    st.exception(erro_load)
    st.stop()

# ─── Função de pré-processamento ─────────────────────────────────────────
def preprocess_image(file):
    conteudo = file.read()
    if len(conteudo) == 0:
        raise ValueError("O arquivo enviado está vazio.")
    try:
        img = Image.open(io.BytesIO(conteudo)).convert("L")
    except UnidentifiedImageError:
        raise ValueError("Arquivo corrompido ou formato inválido.")
    img = img.resize((28, 28))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, (0, -1)).astype(np.float32)

# ─── Função: gera mapa de calor ──────────────────────────────────────────
def gerar_mapa_calor(original, reconstrucao):
    """
    Calcula a diferença pixel a pixel entre original e reconstrução.
    Áreas quentes (vermelho) = maior diferença = modelo não reconheceu bem.
    Áreas frias (azul) = pouca diferença = região familiar ao modelo.
    Retorna a imagem do mapa como objeto PIL para exibir no Streamlit.
    """
    # Diferença absoluta entre original e reconstrução
    diff = np.abs(original[0, :, :, 0] - reconstrucao[0, :, :, 0])

    # Normaliza entre 0 e 1 para o mapa de cores
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    # Aplica o colormap "jet": azul=baixo, verde=médio, vermelho=alto
    colormap   = cm.get_cmap("jet")
    mapa_rgba  = colormap(diff_norm)                        # shape (28,28,4)
    mapa_rgb   = (mapa_rgba[:, :, :3] * 255).astype(np.uint8)  # descarta canal alpha

    # Amplia de 28x28 para 224x224 para ficar visível
    imagem_pil = Image.fromarray(mapa_rgb).resize((224, 224), Image.NEAREST)
    return imagem_pil

# ─── Função: gera link de pesquisa médica ────────────────────────────────
def link_pesquisa(resultado):
    """
    Monta um link clicável para pesquisa no PubMed (base de artigos médicos).
    PubMed é gratuito e contém milhões de artigos científicos.
    """
    if resultado == "PNEUMONIA":
        termo  = "pneumonia+chest+xray+diagnosis"
        descricao = "Pesquisar artigos sobre pneumonia no PubMed"
    else:
        termo  = "normal+chest+xray+findings"
        descricao = "Pesquisar artigos sobre raio-X normal no PubMed"

    url = f"https://pubmed.ncbi.nlm.nih.gov/?term={termo}"
    return url, descricao

# ─── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Pacientes")
    st.divider()

    nome = st.text_input(
        "Nome do paciente",
        placeholder="Ex: João Silva",
        value=st.session_state.paciente_atual
    )
    if nome:
        st.session_state.paciente_atual = nome

    st.divider()

    st.markdown("**Limiar de classificação**")
    LIMIAR_MSE = st.slider(
        "MSE",
        min_value=0.01,
        max_value=1.00,
        value=0.50,
        step=0.01,
        help="Abaixo deste valor = NORMAL. Acima = PNEUMONIA."
    )
    st.caption(f"Normal ≤ {LIMIAR_MSE:.2f}  |  Pneumonia > {LIMIAR_MSE:.2f}")

    st.divider()

    if st.session_state.historico:
        st.markdown("**Analisados nesta sessão:**")
        nomes_vistos = []
        for entrada in reversed(st.session_state.historico):
            nome_pac = entrada["paciente"]
            if nome_pac not in nomes_vistos:
                nomes_vistos.append(nome_pac)
                icone = "🟢" if entrada["resultado"] == "NORMAL" else "🔴"
                st.markdown(f"{icone} {nome_pac}")
    else:
        st.caption("Nenhuma análise ainda.")

    st.divider()

    if st.button("Limpar histórico", use_container_width=True):
        st.session_state.historico = []
        st.rerun()

# ─── Abas principais ─────────────────────────────────────────────────────
aba_analise, aba_historico, aba_normal, aba_pneumonia = st.tabs([
    "Análise",
    "Histórico Geral",
    "Normais",
    "Pneumonia"
])

# ════════════════════════════════════════════════════════════════════════
# ABA 1 — ANÁLISE
# ════════════════════════════════════════════════════════════════════════
with aba_analise:
    st.title("Triagem de Raio-X")
    st.divider()

    if not st.session_state.paciente_atual:
        st.warning("Digite o nome do paciente na barra lateral antes de analisar.")

    uploaded_file = st.file_uploader(
        "Selecione a imagem de raio-X",
        type=FORMATOS_ACEITOS,
        help="Formatos aceitos: PNG, JPG, JPEG"
    )

    if uploaded_file:
        col_img, col_resultado = st.columns([1, 1])

        with col_img:
            st.subheader("Imagem enviada")
            st.image(uploaded_file, width=250)

        if st.button("Analisar", type="primary", use_container_width=True):
            if not st.session_state.paciente_atual:
                st.error("Digite o nome do paciente antes de analisar.")
            else:
                with st.spinner("Processando imagem..."):
                    try:
                        x            = preprocess_image(uploaded_file)
                        reconstrucao = vae(x).numpy()
                        mse          = float(np.mean((x - reconstrucao) ** 2))

                        # Classificação
                        if mse <= LIMIAR_MSE:
                            resultado   = "NORMAL"
                            cor         = "green"
                            mensagem    = "Exame dentro do padrão normal."
                            tipo_alerta = "success"
                        else:
                            resultado   = "PNEUMONIA"
                            cor         = "red"
                            mensagem    = "Sinal de pneumonia detectado. Procure um médico."
                            tipo_alerta = "error"

                        # Gera mapa de calor
                        mapa_calor = gerar_mapa_calor(x, reconstrucao)

                        # Link de pesquisa
                        url_pesquisa, desc_pesquisa = link_pesquisa(resultado)

                        # Salva no histórico
                        st.session_state.historico.append({
                            "paciente":  st.session_state.paciente_atual,
                            "resultado": resultado,
                            "mse":       mse,
                            "arquivo":   uploaded_file.name,
                            "limiar":    LIMIAR_MSE,
                        })

                        # ── Exibe resultado ──────────────────────────────
                        with col_resultado:
                            st.subheader("Resultado")

                            st.metric("MSE (erro de reconstrução)", f"{mse:.4f}")
                            st.markdown(
                                f"**Classificação:** "
                                f"<span style='color:{cor}; font-size:1.6em; font-weight:bold'>"
                                f"{resultado}</span>",
                                unsafe_allow_html=True
                            )
                            getattr(st, tipo_alerta)(mensagem)

                            st.divider()

                            # Mapa de calor
                            st.subheader("Mapa de calor")
                            st.image(mapa_calor, width=224)
                            st.caption(
                                "Azul = região familiar ao modelo  |  "
                                "Vermelho = região com maior diferença"
                            )

                            st.divider()

                            # Link de pesquisa médica
                            st.subheader("Pesquisa médica")
                            st.markdown(
                                f"[🔬 {desc_pesquisa}]({url_pesquisa})",
                                unsafe_allow_html=False
                            )
                            st.caption("Fonte: PubMed — base de artigos científicos gratuita")

                    except ValueError as e:
                        st.error(f"Erro na imagem: {e}")
                    except Exception as e:
                        st.error("Erro inesperado durante a análise.")
                        st.exception(e)

# ════════════════════════════════════════════════════════════════════════
# FUNÇÃO AUXILIAR — renderiza tabela de histórico
# ════════════════════════════════════════════════════════════════════════
def renderizar_tabela(registros):
    if not registros:
        st.info("Nenhum registro encontrado.")
        return

    cab1, cab2, cab3, cab4, cab5 = st.columns([3, 2, 2, 2, 3])
    cab1.markdown("**Paciente**")
    cab2.markdown("**Resultado**")
    cab3.markdown("**MSE**")
    cab4.markdown("**Limiar usado**")
    cab5.markdown("**Arquivo**")
    st.divider()

    for entrada in reversed(registros):
        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 3])
        col1.write(entrada["paciente"])

        if entrada["resultado"] == "NORMAL":
            col2.success("NORMAL")
        else:
            col2.error("PNEUMONIA")

        col3.write(f"{entrada['mse']:.4f}")
        col4.write(f"{entrada.get('limiar', 0.50):.2f}")
        col5.write(entrada["arquivo"])

# ════════════════════════════════════════════════════════════════════════
# ABA 2 — HISTÓRICO GERAL
# ════════════════════════════════════════════════════════════════════════
with aba_historico:
    st.title("Histórico Geral")

    total     = len(st.session_state.historico)
    normais   = sum(1 for e in st.session_state.historico if e["resultado"] == "NORMAL")
    pneumonia = total - normais

    m1, m2, m3 = st.columns(3)
    m1.metric("Total analisado", total)
    m2.metric("Normais",         normais)
    m3.metric("Pneumonia",       pneumonia)

    st.divider()
    renderizar_tabela(st.session_state.historico)

# ════════════════════════════════════════════════════════════════════════
# ABA 3 — SOMENTE NORMAIS
# ════════════════════════════════════════════════════════════════════════
with aba_normal:
    st.title("Exames Normais")
    registros_normais = [
        e for e in st.session_state.historico if e["resultado"] == "NORMAL"
    ]
    st.metric("Total de normais", len(registros_normais))
    st.divider()
    renderizar_tabela(registros_normais)

# ════════════════════════════════════════════════════════════════════════
# ABA 4 — SOMENTE PNEUMONIA
# ════════════════════════════════════════════════════════════════════════
with aba_pneumonia:
    st.title("Exames com Pneumonia")
    registros_pneumonia = [
        e for e in st.session_state.historico if e["resultado"] == "PNEUMONIA"
    ]
    st.metric("Total com pneumonia", len(registros_pneumonia))
    st.divider()
    renderizar_tabela(registros_pneumonia)