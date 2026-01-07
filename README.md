# Interpretability & Causality — FashionMNIST

Este repositório contém o código desenvolvido para o Trabalho Prático da unidade curricular **Interpretabilidade e Causalidade**, com foco na **avaliação quantitativa de métodos de interpretabilidade** aplicados a modelos de classificação de imagens.

O projeto avalia cinco métodos de interpretabilidade (Grad-CAM, Occlusion, Integrated Gradients, LIME e SHAP) aplicados a um modelo de classificação treinado no dataset **FashionMNIST**, utilizando métricas quantitativas inspiradas na literatura para medir **robustez** e **simplicidade** das explicações.

---

## Dataset

- **FashionMNIST**
- 10 classes
- Imagens em tons de cinzento
- Resolução original: 28×28  
- Para o modelo pré-treinado, as imagens são redimensionadas para 224×224

---

## Estrutura do Repositório

├── main.py # Treino, validação, teste e gravação do modelo
├── config.py # Configurações globais (seeds, paths, hiperparâmetros)
├── model.py # CNN simples (baseline)
├── pretrained_model.py # ResNet18 pré-treinada adaptada a 1 canal
├── data_loader.py # DataLoader para modelo simples
├── data_loader_pretrained.py # DataLoader para modelo pré-treinado
├── interpret_methods.py # Implementação dos métodos de interpretabilidade
├── metrics.py # Métricas Avg-Sensitivity e Complexity
├── evaluate.py # Avaliação quantitativa das explicações
├── plots.py # Geração de gráficos e rankings
└── README.md


---

## Métodos de Interpretabilidade Avaliados

- **Grad-CAM**
- **Occlusion**
- **Integrated Gradients**
- **LIME**
- **SHAP (GradientShap / Captum)**

Todos os métodos são aplicados de forma consistente e com pós-processamento comum (normalização, suavização e threshold por percentil).

---

## Métricas Implementadas

- **Avg-Sensitivity**  
  Mede a robustez das explicações face a pequenas perturbações no input.

- **Complexity**  
  Mede a simplicidade das explicações através da entropia da distribuição de importâncias.

Valores mais baixos indicam melhor desempenho em ambas as métricas.

---

## Pipeline Experimental

1. Treino do modelo de classificação (`main.py`)
2. Geração de explicações para imagens do conjunto de teste
3. Cálculo das métricas quantitativas (`evaluate.py`)
4. Agregação de resultados em CSV
5. Geração automática de gráficos e rankings (`plots.py`)
6. Análise quantitativa e qualitativa (ver relatório)

---

## Como Executar

### 1. Treinar o modelo
python main.py

### 2. Avaliar interpretabilidade
python evaluate.py

### 3. Gerar gráficos e rankings
python plots.py

