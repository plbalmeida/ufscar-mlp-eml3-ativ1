# UFSCar - ML in Production
## Módulo 3 - Engenharia de Machine Learning: Model Serving
## Atividade 1

## Projeto de classificação de Iris com MLServer

Este repositório contém o código e os artefatos para a atividade 1 do módulo 3 de Engenharia de Machine Learning, parte do curso de pós-graduação *ML in Production* da UFSCar. O projeto consiste em um modelo de Machine Learning para classificar flores Iris utilizando o MLServer.

## Estrutura do repositório

```bash
/ufscar-mlp-eml3-ativ1
├── Dockerfile
├── README.md
├── data
│   ├── X_test.csv
│   ├── X_train.csv
│   ├── y_test.csv
│   └── y_train.csv
├── models
│   ├── knc
│   │   ├── iris_knc.joblib
│   │   └── model-settings.json
│   └── rf
│       ├── iris_rf.joblib
│       └── model-settings.json
├── requirements.txt
├── settings.json
└── src
    ├── iris-classifier
    │   ├── knc.py
    │   └── rf.py
    └── iris-data
        └── iris_data.py
```

## Pré-requisitos

É necessário ter o Docker instalado, as instruções de como instalar o Docker: [site oficial do Docker](https://www.docker.com/products/docker-desktop).

## Como Rodar o Projeto

### Clonar o repositório

Clone o repositório para sua máquina local usando:

```bash
git clone https://github.com/plbalmeida/ufscar-mlp-eml3-ativ1.git
```

### Executar os scripts

Executar o seguinte comando:

```bash
python3 src/iris-data/iris_data.py src/iris-classifier/knc.py src/iris-classifier/rf.py
```

É esperado que os arquivos .csv dos dados de treino e teste sejam gravados na pasta `data`, e que os arquivos dos modelos treinados `iris_knc.jolib` e `iris_rf.jolib` estejam em seus respectivos diretórios.

### Fazer o build e run do Docker

Fazer o build da imagem Docker do projeto:

```bash
docker build -t ufscar-mlp-eml3-ativ1 .
```

Rodar o container da imagem do Docker criada:

```bash
docker run -p 8080:8080 ufscar-mlp-eml3-ativ1
```

### Predições com o endpoint dos modelos

Obs.: foi utilizado o `jq` para identar o retorno das requisições. Para instalar o mesmo no Ubuntu/Debian:

```bash
sudo apt-get install jq
```

Para inferir com o modelo `DecisionTreeClassifier`:

```bash
$ curl -X POST "http://localhost:8080/v2/models/iris-classifier-rf/infer" -H "Content-Type: application/json" -d '{
  "inputs": [
    {
      "name": "input-0",
      "shape": [1, 4],
      "datatype": "FP32",
      "data": [5.1, 3.5, 1.4, 0.2]
    }
  ]
}' | jq .
```

O retorno esperado é o seguinte:

```bash
{
  "model_name": "iris-classifier-rf",
  "model_version": "1.0",
  "id": "bb5cc19b-91b0-4ea0-b3f0-e1cf726f3994",
  "parameters": {},
  "outputs": [
    {
      "name": "predict",
      "shape": [
        1,
        1
      ],
      "datatype": "INT64",
      "parameters": {
        "content_type": "np"
      },
      "data": [
        0
      ]
    }
  ]
}
```

Para inferir com o modelo `KNeighborsClassifier`:

```bash
$ curl -X POST "http://localhost:8080/v2/models/iris-classifier-knc/infer" -H "Content-Type: application/json" -d '{
  "inputs": [
    {
      "name": "input-0",
      "shape": [1, 4],
      "datatype": "FP32",
      "data": [5.1, 3.5, 1.4, 0.2]
    }
  ]
}' | jq .
```

O retorno esperado da requisição é o seguinte:

```bash
{
  "model_name": "iris-classifier-knc",
  "model_version": "1.0",
  "id": "424bc5d2-d03d-4323-9266-bc481b99223c",
  "parameters": {},
  "outputs": [
    {
      "name": "predict",
      "shape": [
        1,
        1
      ],
      "datatype": "INT64",
      "parameters": {
        "content_type": "np"
      },
      "data": [
        0
      ]
    }
  ]
}
```



# Contribuições

Contribuições são bem-vindas. Para contribuir, por favor, crie um pull request para revisão.