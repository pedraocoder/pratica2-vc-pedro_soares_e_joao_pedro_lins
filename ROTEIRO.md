# VC 2026.2 — Prática II - Classificação de Dígitos de Hidrômetros com AlexNet, VGG16 e ResNet


## Objetivos

1. Construir um dataset de classificação de dígitos (10 classes) a partir das anotações do dataset WaterMeters.
2. Treinar AlexNet, VGG16 e ResNet18 com *transfer learning* e uma CNN pequena treinada do zero (`smallcnn`).
3. Avaliar e comparar os quatro modelos em desempenho e custo computacional.
4. Analisar criticamente os erros e as limitações do pipeline.

## Arquivos deste repositório

| Arquivo | Uso |
|---|---|
| `README.md` | Este roteiro |
| `requirements.txt` | Dependências |
| `splits.csv` | Divisão **oficial** treino/validação/teste (por imagem). Não altere |
| `README_ENTREGA.md` | Modelo do README que o grupo deve entregar |
| `.gitignore` | Exclui dados, recortes, checkpoints e credenciais |

## Regras

- **Licença do dataset: CC BY-NC-ND 4.0.** Uso não comercial e restrito à disciplina. É proibido versionar imagens do dataset, recortes gerados ou checkpoints. Antes do *commit*, limpe a saída das células que exibem imagens do dataset.
- **Uso de IA:** permitido para apoio na codificação. A análise, as respostas e o README devem ser autorais. Declare no README onde houve apoio de IA.
- **Protocolo único:** os quatro modelos devem ser treinados e avaliados nas mesmas condições (seção *Protocolo de treino*).

---

## Parte 1 — Aquisição dos dados

Baixe o dataset [Yandex.Toloka Water Meters](https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset) pela API do Kaggle e verifique a integridade do download.

**Questão 1.1** — O que contém cada pasta e cada coluna do `data.csv`? Quais pastas servem para treinar um modelo?

**Questão 1.2** — O rótulo disponível é por imagem ou por dígito? O que isso implica para a tarefa?

## Parte 2 — Análise exploratória

Explore as imagens, as máscaras e os rótulos. Caracterize a distribuição das leituras.

**Questão 2.1** — Qual é a fonte mais confiável para o rótulo de cada imagem? Justifique com evidências.

**Questão 2.2** — Qual dígito deve ser a classe majoritária? Por quê?

## Parte 3 — Construção do dataset de dígitos

Requisitos:

- Rótulo de cada imagem com **8 dígitos** (5 inteiros + 3 decimais). Descarte as imagens que não se encaixam nesse formato.
- Visor retificado em uma faixa horizontal de **384×64 px**.
- Faixa dividida em **8 recortes de 48×64 px**, um por dígito.
- Correção de visores na vertical e de cabeça para baixo. Reporte quantas faixas foram corrigidas.
- Divisão conforme o `splits.csv` oficial.
- Recortes salvos em `digits/{split}/{digito}/{image_id}_{posicao}.png` e índice em `digits/index.csv` com as colunas `image_id, photo_name, position, digit, split, crop_path`.

Verificações obrigatórias: **1243** imagens válidas, **1** descartada, **9944** recortes, nenhuma imagem presente em mais de um split.

**Questão 3.1** — Por que a divisão é feita por imagem e não por recorte? Por que todos os grupos usam o mesmo `splits.csv`?

**Questão 3.2** — Quantas faixas foram corrigidas? Identifique e quantifique os tipos de recorte errado que o pipeline produz.

**Questão 3.3** — Que erro a divisão da faixa em larguras iguais introduz? Como você o corrigiria?

## Parte 4 — Treinamento dos quatro modelos

### Protocolo de treino

| Modelo | Pesos | Entrada | Batch (CUDA) | Batch (MPS/CPU) | LR cabeça | LR backbone | Épocas |
|---|---|---|---|---|---|---|---|
| `smallcnn` | do zero | 64×48 | 256 | 128 | 1e-3 | — | 10 |
| `alexnet` | IMAGENET1K_V1 | 128×128 | 128 | 64 | 1e-3 | 1e-4 | 8 |
| `vgg16_bn` | IMAGENET1K_V1 | 128×128 | 32 | 16 | 1e-3 | 1e-5 | 8 |
| `resnet18` | IMAGENET1K_V1 | 128×128 | 128 | 64 | 1e-3 | 1e-4 | 8 |

Comum a todos:

- otimizador `AdamW` e agendador `OneCycleLR`;
- `CrossEntropyLoss(label_smoothing=0.1)`;
- 1 época inicial com o backbone congelado (modelos pré-treinados);
- *early stopping* pelo F1-macro da validação, com paciência 3;
- `seed=42`;
- *augmentation*: `RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))` + `ColorJitter(brightness=0.3, contrast=0.3)`, sem flip.

Alterações são permitidas desde que valham para os quatro modelos e estejam documentadas no README.

Salve o histórico de cada modelo em `history_{modelo}.csv` com as colunas `epoch, train_loss, train_acc, val_loss, val_acc, val_f1, seconds`.

**Questão 4.1** — Por que normalizar as imagens com a média e o desvio padrão do ImageNet?

**Questão 4.2** — Compare o número de parâmetros dos quatro modelos. Onde se concentram os parâmetros da AlexNet e da VGG16?

**Questão 4.3** — Qual a diferença entre treinar só a cabeça e fazer *fine-tuning* completo? Por que congelar o backbone apenas na primeira época?

**Questão 4.4** — Por que não usar flip neste problema?

## Parte 5 — Avaliação

Todas as métricas são calculadas no **conjunto de teste**. A validação serve apenas para *early stopping*.

### `results.csv` (uma linha por modelo)

| Coluna | Definição |
|---|---|
| `model` | nome do modelo |
| `acc_digit` | acurácia por recorte |
| `f1_macro` | F1 macro sobre as 10 classes |
| `acc_reading` | fração de imagens com os 8 dígitos corretos |
| `params_M` | número de parâmetros, em milhões |
| `s_per_epoch` | mediana do tempo das épocas de treino, sem validação |
| `ms_per_image` | inferência com batch 1: mediana de 200 execuções após 20 de *warm-up* |
| `ckpt_MB` | tamanho em disco do melhor checkpoint |
| `best_epoch` | época com o maior F1-macro de validação |

### Figuras e relatórios

- `classification_report` de cada modelo
- `figures/confusion_matrices.png` — matrizes de confusão normalizadas
- `figures/accuracy_by_position.png` — acurácia por posição do dígito (0–7)
- `figures/training_curves.png` — curvas de loss e acurácia (treino × validação)
- `figures/performance_vs_cost.png` — F1-macro × tempo de inferência
- Galeria com os 16 erros mais confiantes de cada modelo (apenas no notebook, saída limpa antes do commit)

**Questão 5.1** — Qual modelo teve o melhor F1-macro? O ranking é o mesmo da acurácia? Por quê?

**Questão 5.2** — Quais pares de classes mais se confundem? Por quê?

**Questão 5.3** — Por que `acc_reading` é muito menor que `acc_digit`?

**Questão 5.4** — Qual modelo você colocaria em um aplicativo de celular? Justifique com números.

**Questão 5.5** — O que a ResNet traz em relação à VGG16 neste problema?

**Questão 5.6** — A `smallcnn` foi competitiva com as redes pré-treinadas? O que isso indica?

## Desafio extra (opcional, até +0,5 na nota)

Escolha um:

- Melhore a divisão da faixa em dígitos e meça o ganho com o mesmo protocolo.
- Substitua a classificação por dígito por um modelo de sequência que leia a faixa inteira e compare `acc_reading`.

---

## Entrega

**O entregável é obrigatoriamente o link para o repositório GitHub do grupo**, enviado no Google Classroom até a data informada.

- Um único link por grupo, apontando para o repositório (`https://github.com/<usuario>/<repositorio>`).
- O repositório deve estar acessível ao professor (público ou com o professor adicionado como colaborador).
- Entregas em qualquer outro formato (arquivo anexado, `.zip`, notebook avulso, link do Drive ou do Colab) **não serão aceitas**.
- Será avaliado o estado do repositório no prazo de entrega; commits posteriores não serão considerados.

O repositório deve conter:

- notebook executado (sem imagens do dataset nas saídas);
- `splits.csv` oficial, `results.csv`, `history_*.csv` e a pasta `figures/`;
- `README.md` preenchido a partir do `README_ENTREGA.md`: reprodução, protocolo, resultados, respostas às questões, limitações e declaração de uso de IA.

Não versionar: dataset, recortes, checkpoints, `kaggle.json`.

## Critérios de avaliação

| Item | Peso |
|---|---|
| Aquisição e preparação do dataset (inclui a correção de orientação) | 10% |
| Treinamento dos quatro modelos com protocolo idêntico | 25% |
| Avaliação completa com todas as métricas da Parte 5 | 30% |
| Análise crítica e respostas às questões | 35% |

### Descritores

**Protocolo idêntico (25%)**

| Nível | Descrição |
|---|---|
| Excelente | `splits.csv` oficial nos quatro modelos, seed fixa, hiperparâmetros documentados, execução reproduzível |
| Bom | Protocolo idêntico, sem documentação dos hiperparâmetros |
| Regular | Um modelo com protocolo diferente, sem justificativa |
| Insuficiente | Splits regerados ou recortes da mesma imagem em treino e teste |

**Avaliação (30%)**

| Nível | Descrição |
|---|---|
| Excelente | Todas as métricas no teste, figuras legíveis, `results.csv` coerente com o README |
| Bom | Uma métrica ou figura ausente |
| Regular | Várias métricas ou figuras ausentes |
| Insuficiente | Métricas ausentes ou relatadas sobre a validação |

**Análise (35%)**

| Nível | Descrição |
|---|---|
| Excelente | Respostas fundamentadas nos próprios números, limitações identificadas de forma autônoma, conclusão sobre custo × benefício |
| Bom | Respostas corretas, com referência parcial aos resultados |
| Regular | Respostas parcialmente corretas ou pouco fundamentadas |
| Insuficiente | Respostas genéricas, sem referência aos resultados obtidos |

### Penalidades

| Situação | Penalidade |
|---|---|
| Entrega que não seja o link de um repositório GitHub acessível | Entrega não aceita |
| Vazamento entre splits | Zera o item de protocolo (25%) |
| Uso de modelos sem ser os citados | −30% da nota final |
| Imagens do dataset, recortes ou checkpoints versionados | −10% da nota final |
| Ausência do Roteiro de Uso de IA | −20% da nota final |

---

**Fonte dos dados:** Yandex.Toloka Water Meters Dataset, disponível no Kaggle sob licença CC BY-NC-ND 4.0.
- https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset?resource=download

