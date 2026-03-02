import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mascaras import MASCARAS


def carregar_configuracao(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def correlacao_atrous_numpy(imagem_array, mascara, fator_normalizacao, passo, taxa_dilatacao):
    altura_imagem, largura_imagem, _ = imagem_array.shape

    altura_mascara = len(mascara)
    largura_mascara = len(mascara[0])

    altura_efetiva = (altura_mascara - 1) * taxa_dilatacao + 1
    largura_efetiva = (largura_mascara - 1) * taxa_dilatacao + 1

    altura_saida = (altura_imagem - altura_efetiva) // passo + 1
    largura_saida = (largura_imagem - largura_efetiva) // passo + 1

    if altura_saida <= 0 or largura_saida <= 0:
        raise ValueError(
            "Imagem pequena demais para essa combinacao de filtro/r/stride."
        )

    saida_array = np.zeros((altura_saida, largura_saida, 3), dtype=np.float32)

    y_base = np.arange(altura_saida) * passo
    x_base = np.arange(largura_saida) * passo

    for j in range(altura_mascara):
        for i in range(largura_mascara):
            peso = mascara[j][i]
            if peso == 0:
                continue

            y_indices = y_base + j * taxa_dilatacao
            x_indices = x_base + i * taxa_dilatacao
            saida_array += (
                imagem_array[y_indices[:, None], x_indices[None, :], :] * peso
            )

    if fator_normalizacao:
        saida_array /= fator_normalizacao

    return saida_array


def aplicar_mascara_atrous_sem_numpy(
    imagem_rgb,
    mascara,
    fator_normalizacao,
    passo,
    taxa_dilatacao,
    ativacao,
    nome_mascara=None,
):
    largura_imagem, altura_imagem = imagem_rgb.size
    pixels_entrada = imagem_rgb.load()

    altura_mascara = len(mascara)
    largura_mascara = len(mascara[0])

    altura_efetiva = (altura_mascara - 1) * taxa_dilatacao + 1
    largura_efetiva = (largura_mascara - 1) * taxa_dilatacao + 1

    altura_saida = (altura_imagem - altura_efetiva) // passo + 1
    largura_saida = (largura_imagem - largura_efetiva) // passo + 1

    if altura_saida <= 0 or largura_saida <= 0:
        raise ValueError(
            "Imagem pequena demais para essa combinacao de filtro/r/stride."
        )

    saida_array = [
        [[0.0, 0.0, 0.0] for _ in range(largura_saida)] for _ in range(altura_saida)
    ]

    for y_saida in range(altura_saida):
        y_base = y_saida * passo

        for x_saida in range(largura_saida):
            x_base = x_saida * passo
            acumulado = [0.0, 0.0, 0.0]

            for j in range(altura_mascara):
                for i in range(largura_mascara):
                    peso = mascara[j][i]
                    if peso == 0:
                        continue

                    y_in = y_base + j * taxa_dilatacao
                    x_in = x_base + i * taxa_dilatacao
                    r, g, b = pixels_entrada[x_in, y_in]
                    acumulado[0] += r * peso
                    acumulado[1] += g * peso
                    acumulado[2] += b * peso

            if fator_normalizacao:
                acumulado[0] /= fator_normalizacao
                acumulado[1] /= fator_normalizacao
                acumulado[2] /= fator_normalizacao

            saida_array[y_saida][x_saida] = acumulado

    if nome_mascara and "sobel" in nome_mascara.lower():
        min_val = float("inf")
        max_val = float("-inf")

        for y_saida in range(altura_saida):
            for x_saida in range(largura_saida):
                for c in range(3):
                    valor = abs(saida_array[y_saida][x_saida][c])
                    saida_array[y_saida][x_saida][c] = valor
                    if valor < min_val:
                        min_val = valor
                    if valor > max_val:
                        max_val = valor

        if max_val > min_val:
            escala = 255.0 / (max_val - min_val)
            for y_saida in range(altura_saida):
                for x_saida in range(largura_saida):
                    for c in range(3):
                        saida_array[y_saida][x_saida][c] = (
                            saida_array[y_saida][x_saida][c] - min_val
                        ) * escala
        else:
            for y_saida in range(altura_saida):
                for x_saida in range(largura_saida):
                    saida_array[y_saida][x_saida] = [0.0, 0.0, 0.0]
    elif ativacao == "relu":
        for y_saida in range(altura_saida):
            for x_saida in range(largura_saida):
                for c in range(3):
                    if saida_array[y_saida][x_saida][c] < 0:
                        saida_array[y_saida][x_saida][c] = 0.0

    pixels_saida = []
    for y_saida in range(altura_saida):
        for x_saida in range(largura_saida):
            r, g, b = saida_array[y_saida][x_saida]
            r = int(max(0, min(255, r)))
            g = int(max(0, min(255, g)))
            b = int(max(0, min(255, b)))
            pixels_saida.append((r, g, b))

    imagem_saida = Image.new("RGB", (largura_saida, altura_saida))
    imagem_saida.putdata(pixels_saida)
    return imagem_saida


def aplicar_mascara_atrous(
    imagem_rgb,
    mascara,
    fator_normalizacao,
    passo,
    taxa_dilatacao,
    ativacao,
    nome_mascara=None,
):
    imagem_array = np.asarray(imagem_rgb, dtype=np.float32)
    saida_array = correlacao_atrous_numpy(
        imagem_array, mascara, fator_normalizacao, passo, taxa_dilatacao
    )

    if nome_mascara and "sobel" in nome_mascara.lower():
        saida_array = np.abs(saida_array)
        min_val = saida_array.min()
        max_val = saida_array.max()
        if max_val > min_val:
            saida_array = (saida_array - min_val) / (max_val - min_val) * 255
        else:
            saida_array = np.zeros_like(saida_array)
    elif ativacao == "relu":
        saida_array = np.maximum(saida_array, 0)

    saida_array = np.clip(saida_array, 0, 255).astype(np.uint8)
    return Image.fromarray(saida_array, mode="RGB")


def processar_arquivo(
    input_path,
    output_path,
    mascara_nome,
    passo,
    taxa_dilatacao,
    ativacao,
    custom_mask_data=None,
):
    if mascara_nome not in MASCARAS:
        raise ValueError(f"Mascara invalida: {mascara_nome}")
    if not (1 <= passo <= 5):
        raise ValueError("Passo deve estar entre 1 e 5")
    if not (1 <= taxa_dilatacao <= 5):
        raise ValueError("Taxa de dilatacao deve estar entre 1 e 5")
    if ativacao not in {"relu", "identidade"}:
        raise ValueError("Ativacao deve ser 'relu' ou 'identidade'")

    with Image.open(input_path) as img:
        imagem_rgb = img.convert("RGB")
        if mascara_nome == "sobel_xy":
            imagem_array = np.asarray(imagem_rgb, dtype=np.float32)
            gx = correlacao_atrous_numpy(
                imagem_array,
                MASCARAS["sobel_x"]["mascara"],
                MASCARAS["sobel_x"]["fator_normalizacao"],
                passo,
                taxa_dilatacao,
            )
            gy = correlacao_atrous_numpy(
                imagem_array,
                MASCARAS["sobel_y"]["mascara"],
                MASCARAS["sobel_y"]["fator_normalizacao"],
                passo,
                taxa_dilatacao,
            )
            saida_array = np.sqrt(gx**2 + gy**2)
            min_val = saida_array.min()
            max_val = saida_array.max()
            if max_val > min_val:
                saida_array = (saida_array - min_val) / (max_val - min_val) * 255
            else:
                saida_array = np.zeros_like(saida_array)
            saida_array = np.clip(saida_array, 0, 255).astype(np.uint8)
            nova_imagem = Image.fromarray(saida_array, mode="RGB")
        else:
            mascara = MASCARAS[mascara_nome]
            if mascara_nome == "customizada" and custom_mask_data is not None:
                mascara = custom_mask_data
            nova_imagem = aplicar_mascara_atrous(
                imagem_rgb,
                mascara["mascara"],
                mascara["fator_normalizacao"],
                passo,
                taxa_dilatacao,
                ativacao,
                nome_mascara=mascara_nome,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nova_imagem.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input_path", help="Arquivo de entrada")
    parser.add_argument("--output", dest="output_path", help="Arquivo de saida")
    parser.add_argument("--mascara", help="Nome da mascara")
    parser.add_argument("--passo", type=int, help="Stride")
    parser.add_argument("--taxa-dilatacao", type=int, help="Taxa r")
    parser.add_argument("--ativacao", choices=["relu", "identidade"], help="Ativacao")
    parser.add_argument(
        "--custom-mask-json",
        help="Mascara customizada em JSON: {'mascara': [[...]], 'fator_normalizacao': ...}",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Arquivo de configuracao (usado quando nao passar os parametros acima)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if (
        args.input_path
        and args.output_path
        and args.mascara
        and args.passo
        and args.taxa_dilatacao
        and args.ativacao
    ):
        custom_mask_data = None
        if args.custom_mask_json:
            custom_mask_data = json.loads(args.custom_mask_json)
        processar_arquivo(
            Path(args.input_path),
            Path(args.output_path),
            args.mascara,
            args.passo,
            args.taxa_dilatacao,
            args.ativacao,
            custom_mask_data=custom_mask_data,
        )
        return

    entrada = Path("imagens")
    saida = Path("saida")
    saida.mkdir(exist_ok=True)

    config = carregar_configuracao(args.config)

    for arquivo in entrada.glob("*"):
        print(f"Processando {arquivo.name} com a mascara {config['mascara']}...")
        destino = saida / f"{arquivo.stem}_{config['mascara']}{arquivo.suffix}"
        processar_arquivo(
            arquivo,
            destino,
            config["mascara"],
            int(config["passo"]),
            int(config["taxa_dilatacao"]),
            config["ativacao"],
        )


if __name__ == "__main__":
    main()
