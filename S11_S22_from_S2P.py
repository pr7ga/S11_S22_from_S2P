import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.title("Análise de S11 e S22 a partir de arquivo .S2P")

# Upload do arquivo .s2p
uploaded_file = st.file_uploader("Envie o arquivo .S2P", type=["s2p"])

if uploaded_file:
    # Leitura do arquivo ignorando o cabeçalho
    linhas = uploaded_file.readlines()
    dados = []
    for linha in linhas:
        try:
            partes = linha.decode("utf-8").strip().split()
            if len(partes) == 9:
                dados.append(list(map(float, partes)))
        except:
            continue

    df = pd.DataFrame(dados, columns=[
        "Frequencia", "S11_mag", "S11_fase",
        "S21_mag", "S21_fase", "S12_mag", "S12_fase",
        "S22_mag", "S22_fase"
    ])

    df["S11_dB"] = 20 * np.log10(df["S11_mag"])
    df["S22_dB"] = 20 * np.log10(df["S22_mag"])

    st.success("Arquivo carregado com sucesso!")

    # Entrada dos títulos dos gráficos
    titulo_s11 = st.text_input("Título do gráfico S11", value="S11")
    titulo_s22 = st.text_input("Título do gráfico S22", value="S22")

    # Entrada das três frequências de interesse
    f1 = st.number_input("Frequência 1 (MHz)", value=350.0)
    f2 = st.number_input("Frequência 2 (MHz)", value=400.0)
    f3 = st.number_input("Frequência 3 (MHz)", value=450.0)
    freq_interesse = [f1, f2, f3]

    # Interpolar valores nas frequências de interesse
    def interpola_s_param(df, freq, col):
        return np.interp(freq, df["Frequencia"], df[col])

    resultados = []
    for f in freq_interesse:
        s11_db = interpola_s_param(df, f, "S11_dB")
        s22_db = interpola_s_param(df, f, "S22_dB")
        resultados.append({"Frequência (MHz)": f, "S11 (dB)": s11_db, "S22 (dB)": s22_db})
    resultados_df = pd.DataFrame(resultados)

    # --- Gráfico S11 ---
    fig1, ax1 = plt.subplots()
    ax1.plot(df["Frequencia"], df["S11_dB"], label="S11 (dB)")
    ax1.scatter(resultados_df["Frequência (MHz)"], resultados_df["S11 (dB)"], color='red', label="Frequências de interesse")
    for _, row in resultados_df.iterrows():
        ax1.text(row["Frequência (MHz)"], row["S11 (dB)"], f"{row['S11 (dB)']:.2f} dB", fontsize=8, ha='left', va='bottom')
    ax1.set_xlabel("Frequência (MHz)")
    ax1.set_ylabel("S11 (dB)")
    ax1.set_title(titulo_s11)
    ax1.grid(True)
    ax1.legend()

    # --- Gráfico S22 ---
    fig2, ax2 = plt.subplots()
    ax2.plot(df["Frequencia"], df["S22_dB"], label="S22 (dB)", color='orange')
    ax2.scatter(resultados_df["Frequência (MHz)"], resultados_df["S22 (dB)"], color='red', label="Frequências de interesse")
    for _, row in resultados_df.iterrows():
        ax2.text(row["Frequência (MHz)"], row["S22 (dB)"], f"{row['S22 (dB)']:.2f} dB", fontsize=8, ha='left', va='bottom')
    ax2.set_xlabel("Frequência (MHz)")
    ax2.set_ylabel("S22 (dB)")
    ax2.set_title(titulo_s22)
    ax2.grid(True)
    ax2.legend()

    # Exibição dos gráficos
    st.pyplot(fig1)
    st.pyplot(fig2)

    # Mostrar tabela de resultados
    st.subheader("Valores nas frequências de interesse")
    st.dataframe(resultados_df)

    # --- Download dos dados em CSV ---
    csv = resultados_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar dados em CSV",
        data=csv,
        file_name=f"{titulo_s11}_{titulo_s22}_dados.csv",
        mime="text/csv"
    )

    # --- Download dos gráficos em PNG ---
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", bbox_inches="tight")
    buf1.seek(0)

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches="tight")
    buf2.seek(0)

    st.download_button(
        label="📸 Baixar gráfico S11 (PNG)",
        data=buf1,
        file_name=f"{titulo_s11}.png",
        mime="image/png"
    )

    st.download_button(
        label="📸 Baixar gráfico S22 (PNG)",
        data=buf2,
        file_name=f"{titulo_s22}.png",
        mime="image/png"
    )
