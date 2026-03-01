import subprocess
import sys
import uuid
from pathlib import Path
from datetime import datetime
from time import perf_counter

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "saida"
SAMPLES_DIR = BASE_DIR / "imagens"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
MASCARAS = [
    "gaussiano_5x5",
    "box_1x10",
    "box_10x1",
    "box_10x10",
    "sobel_x",
    "sobel_y",
    "sobel_xy",
]

app = Flask(__name__)


def arquivo_permitido(nome):
    return "." in nome and nome.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def garantir_png_para_preview(caminho_arquivo: Path) -> Path:
    if caminho_arquivo.suffix.lower() == ".png":
        return caminho_arquivo

    caminho_png = caminho_arquivo.with_suffix(".png")
    with Image.open(caminho_arquivo) as img:
        img.convert("RGB").save(caminho_png, format="PNG")
    return caminho_png


def listar_amostras():
    return sorted(
        [
            p.name
            for p in SAMPLES_DIR.iterdir()
            if p.is_file() and arquivo_permitido(p.name)
        ]
    )


@app.route("/preview-upload", methods=["POST"])
def preview_upload():
    arquivo = request.files.get("imagem")
    if not arquivo or arquivo.filename == "":
        return jsonify({"erro": "Selecione uma imagem."}), 400
    if not arquivo_permitido(arquivo.filename):
        return jsonify({"erro": "Formato nao suportado."}), 400

    base_nome = secure_filename(arquivo.filename)
    sufixo = uuid.uuid4().hex[:8]
    nome_input = f"{Path(base_nome).stem}_{sufixo}{Path(base_nome).suffix.lower()}"
    input_path = UPLOAD_DIR / nome_input
    arquivo.save(input_path)

    try:
        preview_path = garantir_png_para_preview(input_path)
    except Exception:
        return jsonify({"erro": "Nao foi possivel gerar preview."}), 400

    return jsonify({"preview_url": f"/uploads/{preview_path.name}", "input_name": nome_input})


@app.route("/", methods=["GET", "POST"])
def index():
    amostras = listar_amostras()
    amostras_set = set(amostras)
    erro = None
    original_url = None
    resultado_url = None
    preview_url = None
    input_name_cache = None
    tempo_processamento_ms = None
    form_data = {
        "amostra": "",
        "mascara": "gaussiano_5x5",
        "passo": "1",
        "taxa_dilatacao": "1",
        "ativacao": "relu",
    }

    if request.method == "POST":
        arquivo = request.files.get("imagem")
        preview_url = request.form.get("preview_url_cache") or None
        input_name_cache = request.form.get("input_name_cache") or None
        amostra = request.form.get("amostra", form_data["amostra"])
        mascara = request.form.get("mascara", form_data["mascara"])
        passo = request.form.get("passo", form_data["passo"])
        taxa = request.form.get("taxa_dilatacao", form_data["taxa_dilatacao"])
        ativacao = request.form.get("ativacao", form_data["ativacao"])
        form_data.update(
            {
                "amostra": amostra,
                "mascara": mascara,
                "passo": passo,
                "taxa_dilatacao": taxa,
                "ativacao": ativacao,
            }
        )

        if arquivo and arquivo.filename and not arquivo_permitido(arquivo.filename):
            erro = "Formato nao suportado."
        elif amostra and amostra not in amostras_set:
            erro = "Imagem de amostra invalida."
        elif mascara not in MASCARAS:
            erro = "Mascara invalida."
        elif (not arquivo or not arquivo.filename) and not input_name_cache and not amostra:
            erro = "Selecione uma imagem."
        else:
            try:
                passo_i = int(passo)
                taxa_i = int(taxa)
                if not (1 <= passo_i <= 5 and 1 <= taxa_i <= 5):
                    raise ValueError
            except ValueError:
                erro = "Passo e taxa de dilatacao devem estar entre 1 e 5."

        if erro is None:
            if arquivo and arquivo.filename:
                base_nome = secure_filename(arquivo.filename)
                sufixo = uuid.uuid4().hex[:8]
                nome_input = f"{Path(base_nome).stem}_{sufixo}{Path(base_nome).suffix.lower()}"
                input_path = UPLOAD_DIR / nome_input
                arquivo.save(input_path)
                input_name_cache = nome_input
            elif amostra:
                input_path = SAMPLES_DIR / amostra
                if not input_path.exists():
                    erro = "Imagem de amostra nao encontrada."
                preview_url = f"/imagens/{input_path.name}"
            else:
                input_path = UPLOAD_DIR / input_name_cache
                if not input_path.exists():
                    erro = "Imagem anterior nao encontrada. Selecione novamente."

            if erro is None:
                timestamp_saida = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                nome_output = (
                    f"{timestamp_saida}_{input_path.stem}_{mascara}"
                    f"{input_path.suffix.lower()}"
                )
                output_path = OUTPUT_DIR / nome_output

                cmd = [
                    sys.executable,
                    str(BASE_DIR / "main.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--mascara",
                    mascara,
                    "--passo",
                    str(passo_i),
                    "--taxa-dilatacao",
                    str(taxa_i),
                    "--ativacao",
                    ativacao,
                ]

                inicio = perf_counter()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                tempo_processamento_ms = (perf_counter() - inicio) * 1000
                if proc.returncode != 0:
                    erro = f"Erro ao processar imagem: {proc.stderr.strip() or proc.stdout.strip()}"
                else:
                    output_preview = garantir_png_para_preview(output_path)
                    if amostra:
                        original_url = f"/imagens/{input_path.name}"
                    else:
                        input_preview = garantir_png_para_preview(input_path)
                        original_url = f"/uploads/{input_preview.name}"
                    resultado_url = f"/saida/{output_preview.name}"
                    preview_url = original_url

    return render_template(
        "index.html",
        mascaras=MASCARAS,
        amostras=amostras,
        form_data=form_data,
        erro=erro,
        preview_url=preview_url,
        input_name_cache=input_name_cache,
        original_url=original_url,
        resultado_url=resultado_url,
        tempo_processamento_ms=tempo_processamento_ms,
    )


@app.route("/uploads/<path:nome>")
def upload_file(nome):
    return send_from_directory(UPLOAD_DIR, nome)


@app.route("/saida/<path:nome>")
def output_file(nome):
    return send_from_directory(OUTPUT_DIR, nome)


@app.route("/imagens/<path:nome>")
def sample_file(nome):
    return send_from_directory(SAMPLES_DIR, nome)


if __name__ == "__main__":
    app.run(debug=True)
