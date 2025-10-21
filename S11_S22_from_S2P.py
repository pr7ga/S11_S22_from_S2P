import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# ==========================
# Função de leitura inteligente de .S2P
# ==========================
def read_s2p_smart(file):
    """
    Lê um arquivo .S2P e retorna DataFrame com colunas:
    'Freq_MHz', 'S11_dB', 'S22_dB'
    Detecta automaticamente se os S estão em dB (DB) ou magnitude (MA)
    e se freq está em Hz.
    """
    raw = file.getvalue().decode(errors="replace").splitlines()

    option_line = None
    data_lines = []
    for ln in raw:
        ln_strip = ln.strip()
        if ln_strip == "" or ln_strip.startswith("!"):
            continue
        if ln_strip.startswith("#"):
            option_line = ln_strip
            continue
        data_lines.append(ln_strip)

    if option_line is None:
        option_line = "# Hz S DB R 50"

    opt = option_line.upper()
    freq_unit_is_hz = "HZ" in opt and "MHZ" not in opt
    s_in_db = "DB" in opt or "DBS" in opt

    cols = []
    for ln in data_lines:
        parts = ln.split()
        if len(parts) < 9:
            continue
        cols.append(parts[:9])

    df = pd.DataFrame(cols, columns=[
        "Freq","S11_val","S11_phase","S21_val","S21_phase",
        "S12_val","S12_phase","S22_val","S22_phase"
    ], dtype=float)

    # Converte frequência para MHz se necessário
    df["Freq_MHz"] = df["Freq"] / 1e6 if freq_unit_is_hz else df["Freq"]

    # Converte para dB se necessário
    if s_in_db:
        df["S11_dB"] = df["S11_val"]
        df["S22_dB"] = df["S22_val"]
    else:
        df["S11_dB"] = 20 * np.log10(np.maximum(df["S11_val"], 1e-20))
        df["S22_dB"] = 20 * np.log10(np.maximum(df["S22_val"], 1e-20))

    return df[["Freq_MHz", "S11_dB", "S22_dB"]]


# ==========================
# Interface Streamlit
# ==========================
st.title("📡 Análise de S11 e S22 a partir de arquivo .S2P")

uploaded_file = st.file_uploader("Envie o arquivo .S2P", type=["s2p"])

if uploaded_file:
    df = read_s2p_smart(uploaded_file)

    st.success("✅ Arquivo lido com sucesso!")

    # Entrada de títulos e frequências
    titulo_s11 = st.text_input("Título do gráfico S11", value="S11")
    titulo_s22 = st.text_input("Título do gráfico S22", value="S22")

    f1 = st.number_input("Frequência 1 (MHz)", value=350.0)
    f2 = st.number_input("Frequência 2 (MHz)", value=400.0)
    f3 = st.number_input("Frequência 3 (MHz)", value=450.0)
    freq_interesse = [f1, f2, f3]

    # Interpolar valores
    def interpola(df, freq, col):
        return np.interp(freq, df["Freq_MHz"], df[col])

    resultados = []
    for f in freq_interesse:
        s11_db = interpola(df, f, "S11_dB")
        s22_db = interpola(df, f, "S22_dB")
        resultados.append({"Frequência (MHz)": f, "S11 (dB)": s11_db, "S22 (dB)": s22_db})
    resultados_df = pd.DataFrame(resultados)

    # ==========================
    # Gráfico S11
    # ==========================
    fig1, ax1 = plt.subplots()
    ax1.plot(df["Freq_MHz"], df["S11_dB"], label="S11 (dB)")
    ax1.scatter(resultados_df["Frequência (MHz)"], resultados_df["S11 (dB)"], color='red', label="Frequências de interesse")
    for _, row in resultados_df.iterrows():
        ax1.text(row["Frequência (MHz)"], row["S11 (dB)"], f"{row['S11 (dB)']:.2f} dB", fontsize=8, ha='left', va='bottom')
    ax1.set_xlabel("Frequência (MHz)")
    ax1.set_ylabel("S11 (dB)")
    ax1.set_title(titulo_s11)
    ax1.grid(True)
    ax1.legend()

    # ==========================
    # Gráfico S22
    # ==========================
    fig2, ax2 = plt.subplots()
    ax2.plot(df["Freq_MHz"], df["S22_dB"], label="S22 (dB)", color='orange')
    ax2.scatter(resultados_df["Frequência (MHz)"], resultados_df["S22 (dB)"], color='red', label="Frequências de interesse")
    for _, row in resultados_df.iterrows():
        ax2.text(row["Frequência (MHz)"], row["S22 (dB)"], f"{row['S22 (dB)']:.2f} dB", fontsize=8, ha='left', va='bottom')
    ax2.set_xlabel("Frequência (MHz)")
    ax2.set_ylabel("S22 (dB)")
    ax2.set_title(titulo_s22)
    ax2.grid(True)
    ax2.legend()

    # Mostrar gráficos
    st.pyplot(fig1)
    st.pyplot(fig2)

    # Mostrar tabela
    st.subheader("📊 Valores nas frequências de interesse")
    st.dataframe(resultados_df)

    # ==========================
    # Download dos resultados
    # ==========================
    csv = resultados_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar dados em CSV",
        data=csv,
        file_name=f"{titulo_s11}_{titulo_s22}_dados.csv",
        mime="text/csv"
    )

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", bbox_inches="tight")
    buf1.seek(0)
    st.download_button(
        label="📸 Baixar gráfico S11 (PNG)",
        data=buf1,
        file_name=f"{titulo_s11}.png",
        mime="image/png"
    )

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches="tight")
    buf2.seek(0)
    st.download_button(
        label="📸 Baixar gráfico S22 (PNG)",
        data=buf2,
        file_name=f"{titulo_s22}.png",
        mime="image/png"
    )
