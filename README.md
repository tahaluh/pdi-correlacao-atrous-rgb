# PDI - Correlação Atrous

## Rodar o app.py (interface web)

1. Crie e ative o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instale as dependencias:

```bash
pip install -r requirements.txt
```

3. Inicie o servidor Flask:

```bash
python3 app.py
```

4. Abra no navegador:

`http://127.0.0.1:5000`

Na página você pode:
- fazer upload da imagem
- escolher máscara, passo, taxa de dilatação e ativação
- processar e visualizar o resultado

A imagem processada é salva em `saida/`.

## Rodar pelo script diretamente

```bash
python main.py --input uploads/exemplo.png --output saida/exemplo_saida.png --mascara sobel_x --passo 1 --taxa-dilatacao 1 --ativacao relu
```

Sem argumentos, `main.py` mantém o modo antigo: lê `config.json` e processa `imagens/*`.
