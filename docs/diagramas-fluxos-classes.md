# Diagramas de Fluxos e Classes

## 1. Visao Geral

Este documento apresenta diagramas baseados no Documento de Requisitos de Software, Historias de Usuario e Casos de Uso do ConvNetJS.

Os diagramas usam sintaxe Mermaid para facilitar visualizacao em ferramentas compativeis com Markdown.

## 2. Diagrama Geral de Casos de Uso

```mermaid
flowchart LR
  Dev[Desenvolvedor JavaScript]
  Est[Estudante/Pesquisador]
  DemoUser[Usuario de demo]
  Mant[Mantenedor]
  Int[Integrador]

  UC01[UC-01 Definir rede neural]
  UC02[UC-02 Treinar modelo]
  UC03[UC-03 Executar inferencia]
  UC04[UC-04 Usar CNN com imagem]
  UC05[UC-05 Serializar e restaurar rede]
  UC06[UC-06 Executar demo no navegador]
  UC07[UC-07 Compilar biblioteca]
  UC08[UC-08 Rodar testes Jasmine]
  UC09[UC-09 Experimentar Deep Q Learning]

  Dev --> UC01
  Dev --> UC02
  Dev --> UC03
  Dev --> UC04
  Int --> UC05
  Int --> UC03
  DemoUser --> UC06
  Mant --> UC07
  Mant --> UC08
  Est --> UC06
  Est --> UC09

  UC01 --> UC02
  UC01 --> UC03
  UC04 --> UC03
  UC02 --> UC05
  UC05 --> UC03
```

## 3. Fluxo UC-01 - Definir Rede Neural

```mermaid
flowchart TD
  A[Inicio] --> B[Instanciar convnetjs.Net]
  B --> C[Criar lista layer_defs]
  C --> D{Primeira camada e input?}
  D -- Nao --> E[Indicar configuracao invalida]
  E --> Z[Fim]
  D -- Sim --> F[Adicionar camadas intermediarias]
  F --> G[Adicionar camada final softmax, svm ou regression]
  G --> H[Chamar net.makeLayers layer_defs]
  H --> I{Camadas reconhecidas?}
  I -- Nao --> J[Indicar tipo de camada invalido]
  J --> Z
  I -- Sim --> K[Calcular dimensoes e montar sequencia interna]
  K --> L[Rede pronta para treino ou inferencia]
  L --> Z
```

## 4. Fluxo UC-02 - Treinar Modelo

```mermaid
flowchart TD
  A[Inicio] --> B[Receber rede criada]
  B --> C[Configurar Trainer ou SGDTrainer]
  C --> D[Preparar Vol de entrada]
  D --> E[Informar alvo esperado]
  E --> F[Executar trainer.train x y]
  F --> G[Executar forward em modo treinamento]
  G --> H[Executar backward e calcular gradientes]
  H --> I{Batch completo?}
  I -- Nao --> J[Acumular gradientes]
  J --> M[Retornar metricas de treino]
  I -- Sim --> K[Aplicar otimizador nos parametros]
  K --> L[Zerar gradientes acumulados]
  L --> M
  M --> N[Modelo atualizado]
  N --> O[Fim]
```

## 5. Fluxo UC-03 - Executar Inferencia

```mermaid
flowchart TD
  A[Inicio] --> B[Receber rede criada]
  B --> C[Receber Vol de entrada]
  C --> D[Chamar net.forward x]
  D --> E[Propagar entrada pela camada input]
  E --> F[Propagar pelas camadas intermediarias]
  F --> G[Propagar pela camada final]
  G --> H[Retornar Vol de saida]
  H --> I{Ultima camada e softmax?}
  I -- Sim --> J[Permitir getPrediction]
  I -- Nao --> K[Usar valores brutos da saida]
  J --> L[Retornar classe de maior probabilidade]
  K --> M[Retornar valores de regressao ou ativacao]
  L --> N[Fim]
  M --> N
```

## 6. Fluxo UC-04 - Usar CNN com Imagem

```mermaid
flowchart TD
  A[Inicio] --> B[Carregar biblioteca no navegador]
  B --> C[Obter elemento de imagem]
  C --> D[Definir input com largura, altura e profundidade]
  D --> E[Adicionar camadas conv e pool]
  E --> F[Adicionar camada final]
  F --> G[Criar rede com makeLayers]
  G --> H[Converter imagem com img_to_vol]
  H --> I{Dimensoes compativeis?}
  I -- Nao --> J[Ajustar imagem ou configuracao da rede]
  J --> H
  I -- Sim --> K[Executar forward]
  K --> L[Obter saida da CNN]
  L --> M[Fim]
```

## 7. Fluxo UC-05 - Serializar e Restaurar Rede

```mermaid
flowchart TD
  A[Inicio] --> B{Operacao desejada}
  B -- Serializar --> C[Receber rede existente]
  C --> D[Chamar net.toJSON]
  D --> E[Gerar JSON com camadas e parametros]
  E --> F[Aplicacao externa armazena JSON]
  F --> Z[Fim]

  B -- Restaurar --> G[Receber JSON salvo]
  G --> H{JSON possui camadas validas?}
  H -- Nao --> I[Indicar restauracao invalida]
  I --> Z
  H -- Sim --> J[Criar nova instancia de Net]
  J --> K[Chamar net.fromJSON]
  K --> L[Recriar camadas e parametros]
  L --> M[Rede pronta para inferencia ou treino]
  M --> Z
```

## 8. Fluxo UC-06 - Executar Demo no Navegador

```mermaid
flowchart TD
  A[Inicio] --> B[Usuario abre HTML em demo]
  B --> C[Navegador carrega CSS e scripts locais]
  C --> D{Scripts carregados com sucesso?}
  D -- Nao --> E[Demo nao inicializa corretamente]
  E --> Z[Fim]
  D -- Sim --> F[Inicializar rede, dados ou simulacao]
  F --> G[Usuario interage com controles da demo]
  G --> H[Executar treinamento, inferencia ou visualizacao]
  H --> I[Atualizar canvas, texto ou graficos]
  I --> Z
```

## 9. Fluxo UC-07 - Compilar Biblioteca

```mermaid
flowchart TD
  A[Inicio] --> B[Mantenedor altera ou revisa src]
  B --> C[Acessar pasta compile]
  C --> D[Executar build Ant]
  D --> E{Ant e YUI Compressor disponiveis?}
  E -- Nao --> F[Build falha por ambiente incompleto]
  F --> Z[Fim]
  E -- Sim --> G[Concatenar arquivos src em ordem fixa]
  G --> H[Gerar build/convnet.js]
  H --> I[Minificar arquivo gerado]
  I --> J[Gerar build/convnet-min.js]
  J --> K[Validar artefatos de distribuicao]
  K --> Z
```

## 10. Fluxo UC-08 - Rodar Testes Jasmine

```mermaid
flowchart TD
  A[Inicio] --> B[Abrir test/jasmine/SpecRunner.html]
  B --> C[Carregar Jasmine]
  C --> D[Carregar biblioteca ConvNetJS]
  D --> E[Carregar specs]
  E --> F[Executar testes de inicializacao]
  F --> G[Executar testes de forward]
  G --> H[Executar testes de treinamento]
  H --> I[Executar teste de gradiente]
  I --> J{Todos passaram?}
  J -- Sim --> K[Resultado aprovado]
  J -- Nao --> L[Exibir falhas para investigacao]
  K --> Z[Fim]
  L --> Z
```

## 11. Fluxo UC-09 - Experimentar Deep Q Learning

```mermaid
flowchart TD
  A[Inicio] --> B[Abrir demo rldemo.html]
  B --> C[Carregar convnetjs e deepqlearn]
  C --> D[Inicializar ambiente da simulacao]
  D --> E[Inicializar agente e Brain]
  E --> F[Capturar estado do ambiente]
  F --> G[Brain escolhe acao]
  G --> H[Ambiente aplica acao]
  H --> I[Calcular recompensa]
  I --> J[Brain aprende com recompensa]
  J --> K{Continuar simulacao?}
  K -- Sim --> F
  K -- Nao --> L[Fim]
```

## 12. Diagrama de Classes Conceitual

```mermaid
classDiagram
  class convnetjs {
    +REVISION
    +Net
    +Vol
    +Trainer
    +SGDTrainer
    +MagicNet
    +img_to_vol()
    +augment()
  }

  class Net {
    +layers
    +makeLayers(defs)
    +forward(V, is_training)
    +backward(y)
    +getCostLoss(V, y)
    +getParamsAndGrads()
    +getPrediction()
    +toJSON()
    +fromJSON(json)
  }

  class Vol {
    +sx
    +sy
    +depth
    +w
    +dw
    +get(x, y, d)
    +set(x, y, d, v)
    +add(x, y, d, v)
    +get_grad(x, y, d)
    +set_grad(x, y, d, v)
    +add_grad(x, y, d, v)
    +cloneAndZero()
    +clone()
    +addFrom(V)
    +addFromScaled(V, a)
    +setConst(a)
    +toJSON()
    +fromJSON(json)
  }

  class Trainer {
    +net
    +learning_rate
    +l1_decay
    +l2_decay
    +batch_size
    +method
    +momentum
    +train(x, y)
  }

  class Layer {
    <<interface>>
    +layer_type
    +out_sx
    +out_sy
    +out_depth
    +forward(V, is_training)
    +backward(y)
    +getParamsAndGrads()
    +toJSON()
    +fromJSON(json)
  }

  class InputLayer
  class FullyConnLayer
  class ConvLayer
  class PoolLayer
  class DropoutLayer
  class LocalResponseNormalizationLayer
  class ReluLayer
  class SigmoidLayer
  class TanhLayer
  class MaxoutLayer
  class SoftmaxLayer
  class SVMLayer
  class RegressionLayer
  class MagicNet
  class Brain

  convnetjs o-- Net
  convnetjs o-- Vol
  convnetjs o-- Trainer
  convnetjs o-- MagicNet

  Net "1" o-- "*" Layer
  Trainer --> Net
  Net --> Vol
  Layer --> Vol

  Layer <|.. InputLayer
  Layer <|.. FullyConnLayer
  Layer <|.. ConvLayer
  Layer <|.. PoolLayer
  Layer <|.. DropoutLayer
  Layer <|.. LocalResponseNormalizationLayer
  Layer <|.. ReluLayer
  Layer <|.. SigmoidLayer
  Layer <|.. TanhLayer
  Layer <|.. MaxoutLayer
  Layer <|.. SoftmaxLayer
  Layer <|.. SVMLayer
  Layer <|.. RegressionLayer

  FullyConnLayer o-- Vol : filters/biases
  ConvLayer o-- Vol : filters/biases
  MagicNet --> Net
  MagicNet --> Trainer
  Brain --> Net : value_net
  Brain --> Trainer : tdtrainer
```

## 13. Diagrama de Componentes

```mermaid
flowchart TB
  subgraph SRC[src]
    Init[convnet_init.js]
    Util[convnet_util.js]
    Vol[convnet_vol.js e convnet_vol_util.js]
    Layers[convnet_layers_*]
    Net[convnet_net.js]
    Trainers[convnet_trainers.js]
    Magic[convnet_magicnet.js]
    Export[convnet_export.js]
  end

  subgraph BUILD[build]
    Bundle[convnet.js]
    Min[convnet-min.js]
    RL[deepqlearn.js]
    Vis[vis.js]
    BuildUtil[util.js]
  end

  subgraph DEMO[demo]
    Html[Demos HTML]
    DemoJS[Scripts JS das demos]
    CSS[CSS]
  end

  subgraph TEST[test]
    Jasmine[Jasmine 2.0.0]
    Specs[NeuralNetSpec.js]
    Runner[SpecRunner.html]
  end

  subgraph COMPILE[compile]
    Ant[build.xml]
    YUI[yuicompressor-2.4.8.jar]
  end

  Init --> Util
  Util --> Vol
  Vol --> Layers
  Layers --> Net
  Net --> Trainers
  Trainers --> Magic
  Magic --> Export

  Ant --> Bundle
  YUI --> Min
  SRC --> Ant
  Bundle --> Html
  Bundle --> Specs
  RL --> Html
  Vis --> Html
  BuildUtil --> Html
  DemoJS --> Html
  CSS --> Html
  Jasmine --> Runner
  Specs --> Runner
```

## 14. Rastreabilidade dos Diagramas

| Diagrama | Base documental | Requisitos/casos relacionados |
| --- | --- | --- |
| Diagrama geral de casos de uso | Casos de Uso e Historias de Usuario | UC-01 a UC-09, HU-01 a HU-12 |
| Fluxo de definicao de rede | Documento de Requisitos e UC-01 | RF-01, RF-02, RF-03 a RF-08 |
| Fluxo de treinamento | UC-02 | RF-10, RF-11 |
| Fluxo de inferencia | UC-03 | RF-09, RF-12, RF-14 |
| Fluxo de CNN com imagem | UC-04 | RF-04, RF-05, RF-15 |
| Fluxo de serializacao | UC-05 | RF-13 |
| Fluxo de demo | UC-06 | RF-18, RNF-05 |
| Fluxo de build | UC-07 | RF-19, RNF-04 |
| Fluxo de testes | UC-08 | RF-20, RNF-07 |
| Fluxo de Deep Q Learning | UC-09 | RF-21 |
| Diagrama de classes | Requisitos, README e `src/` | RF-01 a RF-17 |
| Diagrama de componentes | Plano de Refatoracao e estrutura do repositorio | RF-16 a RF-20 |

