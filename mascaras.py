MASCARAS = {
    "gaussiano_5x5": {
        "mascara": [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        "fator_normalizacao": 256.0,
    },
    "box_1x10": {"mascara": [[1] * 10], "fator_normalizacao": 10.0},
    "box_10x1": {"mascara": [[1] for _ in range(10)], "fator_normalizacao": 10.0},
    "box_10x10": {
        "mascara": [[1] * 10 for _ in range(10)],
        "fator_normalizacao": 100.0,
    },
    "sobel_x": {
        "mascara": [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        "fator_normalizacao": None,
    },
    "sobel_y": {
        "mascara": [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        "fator_normalizacao": None,
    },
    "sobel_xy": {
        "mascara": [
            [0],
        ],
        "fator_normalizacao": None,
    },
    "customizada": {
        "mascara": [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        "fator_normalizacao": 4.0,
    },
}
