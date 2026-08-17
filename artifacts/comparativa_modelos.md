# Comparativa de modelos (NB02)

Corpus: `catalogo_muestra` (1500 docs) · plantilla `A0` · k=10

## Barrido modelo x contrato x dimension

| modelo          | contrato     | sistema                  |   dim |   precision_at_10 |   recall_at_10 |   mrr_at_10 |   ndcg_at_10 |   bytes_por_vector |   segundos |
|:----------------|:-------------|:-------------------------|------:|------------------:|---------------:|------------:|-------------:|-------------------:|-----------:|
| gemini-2        | sin_contrato | gemini-2 [sin_contrato]  |  3072 |            0.7875 |         0.4155 |      1      |       0.7765 |              12288 |      49.59 |
| gemini-2        | sin_contrato | gemini-2 [sin_contrato]  |  1536 |            0.7875 |         0.4155 |      1      |       0.775  |               6144 |      49.59 |
| gemini-2        | sin_contrato | gemini-2 [sin_contrato]  |   768 |            0.7875 |         0.4155 |      1      |       0.7718 |               3072 |      49.59 |
| gemini-2        | nativo       | gemini-2 [nativo]        |  3072 |            0.7625 |         0.3759 |      1      |       0.7509 |              12288 |      45.41 |
| gemini-2        | nativo       | gemini-2 [nativo]        |  1536 |            0.7625 |         0.3759 |      1      |       0.7496 |               6144 |      45.41 |
| gemini-2        | nativo       | gemini-2 [nativo]        |   768 |            0.775  |         0.3795 |      1      |       0.7478 |               3072 |      45.41 |
| gemini-2        | sin_contrato | gemini-2 [sin_contrato]  |   512 |            0.7875 |         0.4155 |      1      |       0.741  |               2048 |      49.59 |
| gemini-2        | nativo       | gemini-2 [nativo]        |   512 |            0.7375 |         0.3688 |      0.9167 |       0.711  |               2048 |      45.41 |
| gemini-2        | nativo       | gemini-2 [nativo]        |   256 |            0.7125 |         0.3297 |      1      |       0.7101 |               1024 |      45.41 |
| gemini-2        | sin_contrato | gemini-2 [sin_contrato]  |   256 |            0.7375 |         0.3965 |      0.9375 |       0.6843 |               1024 |      49.59 |
| gemini-2        | nativo       | gemini-2 [nativo]        |   128 |            0.675  |         0.3036 |      1      |       0.6648 |                512 |      45.41 |
| granite-311m-r2 | nativo       | granite-311m-r2 [nativo] |   768 |            0.6125 |         0.2554 |      0.9167 |       0.6016 |               3072 |   17489.7  |
| granite-311m-r2 | nativo       | granite-311m-r2 [nativo] |   512 |            0.6375 |         0.2902 |      0.8542 |       0.5893 |               2048 |   17489.7  |
| granite-311m-r2 | nativo       | granite-311m-r2 [nativo] |   256 |            0.6125 |         0.2831 |      0.8229 |       0.5629 |               1024 |   17489.7  |
| gemini-2        | sin_contrato | gemini-2 [sin_contrato]  |   128 |            0.65   |         0.2888 |      0.9    |       0.5575 |                512 |      49.59 |
| jina-v3         | sin_contrato | jina-v3 [sin_contrato]   |   512 |            0.6    |         0.2724 |      0.775  |       0.5515 |               2048 |    5751.93 |
| jina-v3         | sin_contrato | jina-v3 [sin_contrato]   |   768 |            0.5875 |         0.2688 |      0.7639 |       0.5398 |               3072 |    5751.93 |
| jina-v3         | sin_contrato | jina-v3 [sin_contrato]   |  1024 |            0.5875 |         0.2688 |      0.7639 |       0.5344 |               4096 |    5751.93 |
| jina-v3         | sin_contrato | jina-v3 [sin_contrato]   |   128 |            0.6    |         0.2724 |      0.7656 |       0.533  |                512 |    5751.93 |
| jina-v3         | sin_contrato | jina-v3 [sin_contrato]   |   256 |            0.575  |         0.2653 |      0.75   |       0.5329 |               1024 |    5751.93 |
| jina-v3         | nativo       | jina-v3 [nativo]         |   512 |            0.5875 |         0.259  |      0.7188 |       0.5188 |               2048 |    4954.58 |
| granite-311m-r2 | nativo       | granite-311m-r2 [nativo] |   128 |            0.5625 |         0.2588 |      0.75   |       0.5164 |                512 |   17489.7  |
| jina-v3         | nativo       | jina-v3 [nativo]         |   256 |            0.55   |         0.2482 |      0.6667 |       0.5105 |               1024 |    4954.58 |
| jina-v3         | nativo       | jina-v3 [nativo]         |   768 |            0.575  |         0.2661 |      0.6875 |       0.5083 |               3072 |    4954.58 |
| jina-v3         | nativo       | jina-v3 [nativo]         |  1024 |            0.575  |         0.2661 |      0.6875 |       0.5055 |               4096 |    4954.58 |
| jina-v3         | nativo       | jina-v3 [nativo]         |   128 |            0.575  |         0.2756 |      0.6429 |       0.4961 |                512 |    4954.58 |

## Regla D09b aplicada

| modelo          | contrato     | sistema                  |   dim |   precision_at_10 |   recall_at_10 |   mrr_at_10 |   ndcg_at_10 |   bytes_por_vector |   segundos | admisible   |   posicion_regla |
|:----------------|:-------------|:-------------------------|------:|------------------:|---------------:|------------:|-------------:|-------------------:|-----------:|:------------|-----------------:|
| gemini-2        | sin_contrato | gemini-2 [sin_contrato]  |   768 |            0.7875 |         0.4155 |      1      |       0.7718 |               3072 |      49.59 | True        |                1 |
| gemini-2        | sin_contrato | gemini-2 [sin_contrato]  |  1536 |            0.7875 |         0.4155 |      1      |       0.775  |               6144 |      49.59 | True        |                2 |
| gemini-2        | sin_contrato | gemini-2 [sin_contrato]  |  3072 |            0.7875 |         0.4155 |      1      |       0.7765 |              12288 |      49.59 | True        |                3 |
| gemini-2        | nativo       | gemini-2 [nativo]        |   128 |            0.675  |         0.3036 |      1      |       0.6648 |                512 |      45.41 | False       |                4 |
| gemini-2        | sin_contrato | gemini-2 [sin_contrato]  |   128 |            0.65   |         0.2888 |      0.9    |       0.5575 |                512 |      49.59 | False       |                5 |
| jina-v3         | sin_contrato | jina-v3 [sin_contrato]   |   128 |            0.6    |         0.2724 |      0.7656 |       0.533  |                512 |    5751.93 | False       |                6 |
| granite-311m-r2 | nativo       | granite-311m-r2 [nativo] |   128 |            0.5625 |         0.2588 |      0.75   |       0.5164 |                512 |   17489.7  | False       |                7 |
| jina-v3         | nativo       | jina-v3 [nativo]         |   128 |            0.575  |         0.2756 |      0.6429 |       0.4961 |                512 |    4954.58 | False       |                8 |
| gemini-2        | nativo       | gemini-2 [nativo]        |   256 |            0.7125 |         0.3297 |      1      |       0.7101 |               1024 |      45.41 | False       |                9 |
| gemini-2        | sin_contrato | gemini-2 [sin_contrato]  |   256 |            0.7375 |         0.3965 |      0.9375 |       0.6843 |               1024 |      49.59 | False       |               10 |
| granite-311m-r2 | nativo       | granite-311m-r2 [nativo] |   256 |            0.6125 |         0.2831 |      0.8229 |       0.5629 |               1024 |   17489.7  | False       |               11 |
| jina-v3         | sin_contrato | jina-v3 [sin_contrato]   |   256 |            0.575  |         0.2653 |      0.75   |       0.5329 |               1024 |    5751.93 | False       |               12 |
| jina-v3         | nativo       | jina-v3 [nativo]         |   256 |            0.55   |         0.2482 |      0.6667 |       0.5105 |               1024 |    4954.58 | False       |               13 |
| gemini-2        | sin_contrato | gemini-2 [sin_contrato]  |   512 |            0.7875 |         0.4155 |      1      |       0.741  |               2048 |      49.59 | False       |               14 |
| gemini-2        | nativo       | gemini-2 [nativo]        |   512 |            0.7375 |         0.3688 |      0.9167 |       0.711  |               2048 |      45.41 | False       |               15 |
| granite-311m-r2 | nativo       | granite-311m-r2 [nativo] |   512 |            0.6375 |         0.2902 |      0.8542 |       0.5893 |               2048 |   17489.7  | False       |               16 |
| jina-v3         | sin_contrato | jina-v3 [sin_contrato]   |   512 |            0.6    |         0.2724 |      0.775  |       0.5515 |               2048 |    5751.93 | False       |               17 |
| jina-v3         | nativo       | jina-v3 [nativo]         |   512 |            0.5875 |         0.259  |      0.7188 |       0.5188 |               2048 |    4954.58 | False       |               18 |
| gemini-2        | nativo       | gemini-2 [nativo]        |   768 |            0.775  |         0.3795 |      1      |       0.7478 |               3072 |      45.41 | False       |               19 |
| granite-311m-r2 | nativo       | granite-311m-r2 [nativo] |   768 |            0.6125 |         0.2554 |      0.9167 |       0.6016 |               3072 |   17489.7  | False       |               20 |
| jina-v3         | sin_contrato | jina-v3 [sin_contrato]   |   768 |            0.5875 |         0.2688 |      0.7639 |       0.5398 |               3072 |    5751.93 | False       |               21 |
| jina-v3         | nativo       | jina-v3 [nativo]         |   768 |            0.575  |         0.2661 |      0.6875 |       0.5083 |               3072 |    4954.58 | False       |               22 |
| jina-v3         | sin_contrato | jina-v3 [sin_contrato]   |  1024 |            0.5875 |         0.2688 |      0.7639 |       0.5344 |               4096 |    5751.93 | False       |               23 |
| jina-v3         | nativo       | jina-v3 [nativo]         |  1024 |            0.575  |         0.2661 |      0.6875 |       0.5055 |               4096 |    4954.58 | False       |               24 |
| gemini-2        | nativo       | gemini-2 [nativo]        |  1536 |            0.7625 |         0.3759 |      1      |       0.7496 |               6144 |      45.41 | False       |               25 |
| gemini-2        | nativo       | gemini-2 [nativo]        |  3072 |            0.7625 |         0.3759 |      1      |       0.7509 |              12288 |      45.41 | False       |               26 |

## Denso frente al baseline lexico de NB01

| sistema                  | familia   | modelo          | contrato     |   dim |   precision_at_10 |   recall_at_10 |   mrr_at_10 |   ndcg_at_10 |   delta_vs_mejor_lexico |
|:-------------------------|:----------|:----------------|:-------------|------:|------------------:|---------------:|------------:|-------------:|------------------------:|
| gemini-2 [sin_contrato]  | denso     | gemini-2        | sin_contrato |  3072 |            0.7875 |         0.4155 |      1      |       0.7765 |                  0.1253 |
| bm25                     | léxico    |                 |              |       |            0.7125 |         0.313  |      0.8906 |       0.6512 |                  0      |
| granite-311m-r2 [nativo] | denso     | granite-311m-r2 | nativo       |   768 |            0.6125 |         0.2554 |      0.9167 |       0.6016 |                 -0.0496 |
| tfidf                    | léxico    |                 |              |       |            0.6    |         0.2324 |      0.875  |       0.5654 |                 -0.0858 |
| jina-v3 [sin_contrato]   | denso     | jina-v3         | sin_contrato |   512 |            0.6    |         0.2724 |      0.775  |       0.5515 |                 -0.0997 |
