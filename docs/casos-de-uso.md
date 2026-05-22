# Casos de Uso

## 1. Visao Geral

Este documento descreve os principais casos de uso do ConvNetJS, considerando sua natureza de biblioteca JavaScript para redes neurais e seus exemplos estaticos em navegador.

## 2. Atores

| Ator | Descricao |
| --- | --- |
| Desenvolvedor JavaScript | Usa a API da biblioteca para criar, treinar e executar redes neurais. |
| Estudante/Pesquisador | Explora conceitos de Deep Learning e aprendizado por reforco. |
| Usuario de demo | Acessa exemplos HTML/JS prontos no navegador. |
| Mantenedor | Mantem, testa, compila e distribui o projeto. |
| Integrador | Incorpora o bundle ou export CommonJS em outro projeto. |

## 3. Lista de Casos de Uso

| ID | Nome | Ator principal | Requisitos relacionados |
| --- | --- | --- | --- |
| UC-01 | Definir rede neural | Desenvolvedor JavaScript | RF-01, RF-02, RF-03, RF-04, RF-05, RF-06, RF-07, RF-08 |
| UC-02 | Treinar modelo | Desenvolvedor JavaScript | RF-10, RF-11 |
| UC-03 | Executar inferencia | Desenvolvedor JavaScript | RF-09, RF-12, RF-14 |
| UC-04 | Usar CNN com imagem | Desenvolvedor JavaScript | RF-04, RF-05, RF-14, RF-15 |
| UC-05 | Serializar e restaurar rede | Integrador | RF-13 |
| UC-06 | Executar demo no navegador | Usuario de demo | RF-18, RNF-01, RNF-05 |
| UC-07 | Compilar biblioteca | Mantenedor | RF-19, RNF-03, RNF-04 |
| UC-08 | Rodar testes Jasmine | Mantenedor | RF-20, RNF-07 |
| UC-09 | Experimentar Deep Q Learning | Estudante/Pesquisador | RF-21 |

## 4. Especificacao dos Casos de Uso

### UC-01 - Definir rede neural

| Campo | Descricao |
| --- | --- |
| Ator principal | Desenvolvedor JavaScript |
| Objetivo | Criar uma rede neural a partir de uma lista ordenada de camadas. |
| Pre-condicoes | A biblioteca ConvNetJS esta carregada no ambiente de execucao. |
| Pos-condicoes | A instancia de `Net` contem as camadas criadas e esta pronta para inferencia ou treinamento. |

Fluxo principal:

1. O ator instancia `convnetjs.Net`.
2. O ator cria uma lista de definicoes de camadas iniciada por `input`.
3. O ator adiciona camadas intermediarias, como `fc`, `conv`, `pool`, ativacoes, dropout ou normalizacao.
4. O ator adiciona uma camada final adequada, como `softmax`, `svm` ou `regression`.
5. O ator chama `makeLayers` com a lista de camadas.
6. A biblioteca cria a sequencia interna de camadas e calcula dimensoes de entrada e saida.

Fluxos alternativos:

- Se a lista possuir menos de duas camadas, a biblioteca deve indicar configuracao invalida.
- Se a primeira camada nao for `input`, a biblioteca deve indicar configuracao invalida.
- Se uma ativacao suportada for declarada em uma camada, a biblioteca deve inserir a camada de ativacao correspondente.

### UC-02 - Treinar modelo

| Campo | Descricao |
| --- | --- |
| Ator principal | Desenvolvedor JavaScript |
| Objetivo | Ajustar os parametros da rede usando dados de entrada e alvo esperado. |
| Pre-condicoes | Existe uma rede criada e um treinador configurado. |
| Pos-condicoes | Os pesos da rede sao atualizados e podem melhorar a resposta ao padrao treinado. |

Fluxo principal:

1. O ator instancia um treinador, como `SGDTrainer`, associado a uma rede.
2. O ator prepara um volume de entrada.
3. O ator informa o alvo esperado, como classe ou valor de regressao.
4. O ator chama `train`.
5. A biblioteca executa propagacao direta, retropropagacao e atualizacao de parametros.
6. O ator avalia a saida apos o treinamento.

Fluxos alternativos:

- Se parametros de treinamento forem inadequados, o treinamento pode apresentar baixa convergencia.
- Se a rede nao estiver criada corretamente, o treinamento nao deve ser considerado valido.

### UC-03 - Executar inferencia

| Campo | Descricao |
| --- | --- |
| Ator principal | Desenvolvedor JavaScript |
| Objetivo | Obter a saida da rede para uma nova entrada. |
| Pre-condicoes | Existe uma rede criada e um volume de entrada compativel. |
| Pos-condicoes | A biblioteca retorna um volume com probabilidades, valores ou ativacoes finais. |

Fluxo principal:

1. O ator cria ou obtem um `Vol` com dados de entrada.
2. O ator chama `forward` na rede.
3. A biblioteca propaga os dados por todas as camadas.
4. A biblioteca retorna o volume de saida.
5. Se a rede termina em `softmax`, o ator pode chamar `getPrediction`.

Fluxos alternativos:

- Se o volume de entrada nao corresponder as dimensoes esperadas, o resultado pode ser invalido ou ocorrer erro em tempo de execucao.
- Se `getPrediction` for chamado sem camada final `softmax`, a biblioteca deve indicar uso inadequado.

### UC-04 - Usar CNN com imagem

| Campo | Descricao |
| --- | --- |
| Ator principal | Desenvolvedor JavaScript |
| Objetivo | Processar imagem em uma rede convolucional. |
| Pre-condicoes | A biblioteca esta carregada no navegador e existe uma imagem disponivel. |
| Pos-condicoes | A imagem e transformada em volume e processada pela rede. |

Fluxo principal:

1. O ator define uma camada `input` com largura, altura e profundidade da imagem.
2. O ator adiciona camadas convolucionais, pooling e camada final.
3. O ator cria a rede com `makeLayers`.
4. O ator converte a imagem para `Vol` usando utilitario da biblioteca.
5. O ator chama `forward` com o volume gerado.
6. A rede retorna a saida do processamento.

Fluxos alternativos:

- Se a imagem tiver dimensoes diferentes das esperadas, o ator deve redimensionar ou ajustar a configuracao da rede.
- Se a execucao ocorrer fora do navegador, utilitarios dependentes de DOM podem nao estar disponiveis.

### UC-05 - Serializar e restaurar rede

| Campo | Descricao |
| --- | --- |
| Ator principal | Integrador |
| Objetivo | Salvar e recuperar a estrutura e parametros de uma rede. |
| Pre-condicoes | Existe uma rede criada ou um JSON valido previamente gerado. |
| Pos-condicoes | O estado da rede e representado em JSON ou restaurado a partir dele. |

Fluxo principal:

1. O ator chama `toJSON` em uma rede existente.
2. A biblioteca retorna objeto JSON com a representacao das camadas.
3. O ator armazena ou transporta essa representacao conforme sua aplicacao externa.
4. Para restaurar, o ator cria uma nova instancia de rede.
5. O ator chama `fromJSON` com o objeto salvo.
6. A biblioteca recria as camadas e parametros.

Fluxos alternativos:

- Se o JSON estiver incompleto ou corrompido, a restauracao pode falhar.
- A biblioteca nao define mecanismo proprio de persistencia em banco de dados.

### UC-06 - Executar demo no navegador

| Campo | Descricao |
| --- | --- |
| Ator principal | Usuario de demo |
| Objetivo | Interagir com exemplos educacionais do projeto. |
| Pre-condicoes | O navegador consegue abrir os arquivos HTML e scripts locais da pasta `demo/`. |
| Pos-condicoes | O usuario visualiza ou interage com uma demonstracao de redes neurais. |

Fluxo principal:

1. O ator abre um arquivo HTML da pasta `demo/`.
2. O navegador carrega CSS e scripts JavaScript locais.
3. A demo inicializa a rede ou simulacao correspondente.
4. O ator interage com controles, dados ou visualizacoes disponiveis.
5. A demo apresenta resultados, treinamento ou comportamento do modelo.

Fluxos alternativos:

- Se algum script local nao for carregado, a demo pode nao funcionar corretamente.
- Se a demo depender de recursos especificos do navegador, ambientes antigos podem apresentar incompatibilidades.

### UC-07 - Compilar biblioteca

| Campo | Descricao |
| --- | --- |
| Ator principal | Mantenedor |
| Objetivo | Gerar arquivos distribuiveis a partir dos arquivos em `src/`. |
| Pre-condicoes | O ambiente possui Apache Ant e `yuicompressor` conforme esperado pelo script. |
| Pos-condicoes | Os arquivos compilados e minificados sao gerados em `build/`. |

Fluxo principal:

1. O ator altera ou revisa arquivos em `src/`.
2. O ator acessa a pasta `compile/`.
3. O ator executa o comando Ant indicado no README.
4. O script concatena os fontes.
5. O script minifica o resultado.
6. Os artefatos finais sao disponibilizados em `build/`.

Fluxos alternativos:

- Se Ant nao estiver instalado, a compilacao nao sera executada.
- Se o compressor nao estiver disponivel, a minificacao podera falhar.

### UC-08 - Rodar testes Jasmine

| Campo | Descricao |
| --- | --- |
| Ator principal | Mantenedor |
| Objetivo | Verificar comportamento basico da biblioteca apos alteracoes. |
| Pre-condicoes | Os arquivos de teste e bibliotecas Jasmine estao disponiveis em `test/jasmine/`. |
| Pos-condicoes | O mantenedor obtem resultado de sucesso ou falha dos testes. |

Fluxo principal:

1. O ator abre `test/jasmine/SpecRunner.html`.
2. O Jasmine carrega as dependencias e especificacoes.
3. Os testes criam uma rede neural simples.
4. Os testes verificam inicializacao, propagacao direta, treinamento e gradiente.
5. O runner exibe os resultados.

Fluxos alternativos:

- Se scripts da biblioteca nao forem carregados, os testes falham.
- Se uma mudanca alterar comportamento esperado, o runner apresenta falhas para investigacao.

### UC-09 - Experimentar Deep Q Learning

| Campo | Descricao |
| --- | --- |
| Ator principal | Estudante/Pesquisador |
| Objetivo | Observar aprendizado por reforco usando o modulo experimental de Deep Q Learning. |
| Pre-condicoes | A demo de aprendizado por reforco e seus scripts estao disponiveis. |
| Pos-condicoes | O ator observa uma simulacao ou exemplo de agente em treinamento. |

Fluxo principal:

1. O ator abre a demo `rldemo.html`.
2. A pagina carrega o modulo de Deep Q Learning e scripts de apoio.
3. A simulacao e inicializada.
4. O ator observa o comportamento do agente ao longo do tempo.
5. O ator usa a demo como apoio educacional ou experimental.

Fluxos alternativos:

- Por ser experimental, o modulo pode nao oferecer a mesma estabilidade das funcionalidades principais da biblioteca.
- Se a demo nao carregar seus scripts, a simulacao nao sera executada.

## 5. Relacionamento com Historias de Usuario

| Caso de uso | Historias relacionadas |
| --- | --- |
| UC-01 | HU-01, HU-06 |
| UC-02 | HU-02 |
| UC-03 | HU-03, HU-06 |
| UC-04 | HU-05 |
| UC-05 | HU-04 |
| UC-06 | HU-07, HU-12 |
| UC-07 | HU-08 |
| UC-08 | HU-09 |
| UC-09 | HU-10 |

