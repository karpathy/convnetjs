# Historias de Usuario

## 1. Visao Geral

Este documento apresenta historias de usuario para o ConvNetJS, considerando o projeto como uma biblioteca JavaScript de redes neurais com demos estaticas no navegador e suporte a uso em Node.js.

As historias seguem o formato:

> Como [ator], quero [acao], para [beneficio].

## 2. Atores

| Ator | Descricao |
| --- | --- |
| Desenvolvedor JavaScript | Pessoa que integra ou usa a biblioteca em projetos browser ou Node.js. |
| Estudante/Pesquisador | Pessoa que aprende, testa ou demonstra conceitos de Deep Learning. |
| Usuario de demo | Pessoa que interage com exemplos prontos no navegador. |
| Mantenedor | Pessoa que corrige, testa, compila e distribui a biblioteca. |
| Integrador | Pessoa que usa o bundle compilado ou a API CommonJS em outro sistema. |

## 3. Historias

| ID | Historia | Prioridade | Requisitos relacionados |
| --- | --- | --- | --- |
| HU-01 | Como desenvolvedor JavaScript, quero definir uma rede neural por uma lista de camadas, para configurar arquiteturas adequadas ao meu problema. | Alta | RF-01, RF-02, RF-03, RF-04, RF-05, RF-06, RF-07, RF-08 |
| HU-02 | Como estudante/pesquisador, quero treinar uma rede neural com exemplos rotulados, para observar a reducao de erro e o ajuste dos pesos. | Alta | RF-10, RF-11 |
| HU-03 | Como desenvolvedor JavaScript, quero executar inferencia em uma rede treinada, para obter probabilidades, valores de regressao ou predicoes. | Alta | RF-09, RF-12, RF-14 |
| HU-04 | Como integrador, quero serializar e restaurar uma rede em JSON, para salvar e reutilizar modelos entre execucoes. | Alta | RF-13 |
| HU-05 | Como desenvolvedor, quero converter imagens em volumes de entrada, para usar redes convolucionais em problemas visuais. | Media | RF-04, RF-14, RF-15 |
| HU-06 | Como integrador, quero usar a biblioteca tanto no navegador quanto em Node.js, para adapta-la a diferentes ambientes JavaScript. | Alta | RF-16, RF-17, RNF-01, RNF-02 |
| HU-07 | Como usuario de demo, quero abrir exemplos interativos no navegador, para aprender visualmente como os modelos se comportam. | Media | RF-18, RNF-05 |
| HU-08 | Como mantenedor, quero compilar os arquivos de `src/` em uma versao distribuivel, para publicar ou testar alteracoes da biblioteca. | Media | RF-19, RNF-03, RNF-04 |
| HU-09 | Como mantenedor, quero executar testes automatizados no Jasmine, para verificar se inicializacao, inferencia, treino e gradientes continuam funcionando. | Media | RF-20, RNF-07 |
| HU-10 | Como estudante/pesquisador, quero experimentar o modulo de Deep Q Learning, para explorar aprendizado por reforco em JavaScript. | Baixa | RF-21 |
| HU-11 | Como desenvolvedor, quero consultar exemplos simples no README e nas demos, para iniciar o uso da biblioteca com pouco codigo. | Media | RNF-08 |
| HU-12 | Como usuario preocupado com privacidade, quero que as demos funcionem localmente no navegador, para evitar envio obrigatorio de dados a servidores externos. | Media | RNF-05, RNF-06, RNF-11 |

## 4. Criterios de Aceitacao

### HU-01 - Definir rede neural por camadas

- Dado que existe uma lista de definicoes de camadas iniciada por `input`, quando o usuario chama `makeLayers`, entao a rede deve criar as camadas correspondentes.
- Dado que uma camada `fc` ou `conv` declara ativacao, quando a rede e criada, entao a ativacao deve ser incorporada ao fluxo da rede.
- Dado que a lista nao inicia com `input`, quando a rede e criada, entao a biblioteca deve indicar erro de configuracao.

### HU-02 - Treinar rede neural

- Dado que existe uma rede criada e um treinador configurado, quando o usuario chama `train` com entrada e rotulo, entao os parametros devem ser atualizados.
- Dado um exemplo de classificacao simples, quando o treinamento ocorre com taxa adequada, entao a probabilidade da classe correta deve aumentar, conforme comportamento coberto pelos testes existentes.

### HU-03 - Executar inferencia

- Dado que existe uma rede criada, quando o usuario chama `forward` com um `Vol`, entao a biblioteca deve retornar um `Vol` de saida.
- Dado que a ultima camada e `softmax`, quando o usuario chama `getPrediction`, entao a biblioteca deve retornar o indice de maior probabilidade.

### HU-04 - Serializar e restaurar rede

- Dado que existe uma rede criada, quando o usuario chama `toJSON`, entao a biblioteca deve retornar uma representacao JSON das camadas.
- Dado um JSON valido de rede, quando o usuario chama `fromJSON`, entao a rede deve ser reconstruida com suas camadas e parametros.

### HU-05 - Usar imagens como entrada

- Dado um elemento de imagem disponivel no navegador, quando o usuario usa o utilitario de conversao, entao a imagem deve ser transformada em volume compativel com a rede.
- Dado uma rede convolucional configurada para as dimensoes esperadas, quando o volume da imagem e processado, entao a rede deve produzir uma saida.

### HU-06 - Usar no navegador e em Node.js

- Dado que a biblioteca e carregada em pagina web sem CommonJS, entao ela deve ficar disponivel como `convnetjs` no escopo global.
- Dado que a biblioteca e carregada em ambiente CommonJS, entao ela deve ser exportada por `module.exports`.

### HU-07 - Abrir demos interativas

- Dado que o usuario abre uma pagina em `demo/`, quando os scripts locais carregam corretamente, entao a demonstracao deve permitir interacao ou visualizacao do exemplo.
- Dado que nao ha backend no projeto, entao a demo deve operar como pagina estatica.

### HU-08 - Compilar biblioteca

- Dado que o ambiente possui Apache Ant e o compressor esperado, quando o mantenedor executa o script de build, entao os arquivos em `build/` devem ser gerados conforme o processo de concatenacao e minificacao.

### HU-09 - Executar testes

- Dado que o usuario abre o `SpecRunner.html`, quando o Jasmine executa os testes, entao os cenarios de inicializacao, propagacao direta, treinamento e gradiente devem ser avaliados.

### HU-10 - Experimentar Deep Q Learning

- Dado que o usuario abre a demo de aprendizado por reforco, quando a simulacao e executada, entao o modulo experimental deve permitir observar comportamento de agente treinado por Deep Q Learning.

### HU-11 - Consultar exemplos

- Dado que o usuario acessa o README, quando consulta os exemplos de codigo, entao deve conseguir identificar o fluxo minimo de criacao, inferencia e treinamento de uma rede.

### HU-12 - Operar localmente

- Dado que as demos sao arquivos estaticos, quando executadas no navegador, entao nao devem exigir servidor ou banco de dados proprio do projeto para suas funcionalidades principais.

## 5. Priorizacao

| Prioridade | Criterio |
| --- | --- |
| Alta | Capacidade essencial para uso da biblioteca como motor de redes neurais. |
| Media | Capacidade importante para distribuicao, aprendizado, manutencao ou ergonomia. |
| Baixa | Capacidade experimental ou complementar ao nucleo principal. |

