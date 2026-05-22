# Sobre este projeto

## Visão geral do repositório

Esse repositório é o `convnetjs` do Andrej Karpathy. É uma biblioteca JavaScript para redes neurais e convolucionais, com foco em uso no navegador e compatibilidade com Node.js. Não é uma aplicação de servidor com banco de dados; é um motor de Deep Learning escrito em JS.

## Para que serve

- Implementa redes neurais em JavaScript.
- Suporta:
  - camadas totalmente conectadas (`fc`)
  - camadas convolucionais
  - pooling
  - normalização
  - funções de perda como `softmax`, `SVM`, `L2`
  - otimizadores / treinadores como SGD, Adagrad, Adadelta
  - módulo experimental de Reinforcement Learning (Deep Q Learning)
- Inclui demos de exemplos em browser, como MNIST, CIFAR-10, regressão, autoencoder e RL.

## Linguagem e execução

- Linguagem: **JavaScript**
- Execução:
  - browser
  - Node.js (há wrappers `module.exports` em `src/convnet_export.js` e em `build/*`)
- Não há TypeScript.
- Não há `package.json` neste workspace, mas há `bower.json` indicando uso como biblioteca front-end/AMD/Node.

## Arquitetura do software

### Camadas principais

- `src/`
  - Contém a implementação modular do motor de rede neural.
  - Arquivos principais:
    - `convnet_net.js` — rede neural, criação de camadas e fluxo forwards/backwards
    - `convnet_trainers.js` — algoritmos de treinamento
    - `convnet_vol.js` / `convnet_vol_util.js` — tensores / volumes de dados
    - `convnet_util.js` — utilitários gerais
    - `convnet_layers_*` — implementação de camadas e funções de ativação:
      - `dotproducts`, `dropout`, `input`, `loss`, `nonlinearities`, `normalization`, `pool`
    - `convnet_magicnet.js` — rede “mágica” / utilitários de alto nível
    - `convnet_export.js` — exportação para Node.js / CommonJS

### Build e distribuição

- `build/`
  - Contém versões concatenadas/minificadas geradas:
    - `convnet.js`
    - `convnet-min.js`
  - Também há outros módulos compilados como `deepqlearn.js`, `util.js`, `vis.js`
- `compile/`
  - Contém `build.xml` para compilar o projeto com Apache Ant e `yuicompressor`.

### Demos

- `demo/`
  - HTML e JS para demos interativas no browser.
  - Usa arquivos estáticos e exemplos de treinamento/visualização.
  - Não há backend próprio; os demos são páginas estáticas.

## Organização de pastas

- `src/` — código-fonte da biblioteca
- `build/` — saída compilada / bundle JS
- `compile/` — script de build Ant
- `demo/` — demonstrações em HTML/JS/CSS
- `test/` — testes Jasmine
- `bower.json` — metadados do pacote Bower
- `Readme.md` — documentação e exemplos

## Banco de dados e APIs externas

- **Não há banco de dados** neste repositório.
- **Não há tabelas** ou modelagem de dados de banco de dados.
- Não há evidência de integração com nenhuma API externa no código principal.
- As demos usam apenas browser APIs e bibliotecas JS locais.

## Dependências e bibliotecas usadas

### Dependências internas / bundled

- Biblioteca principal: `convnetjs` em `src/`
- Demos utilizam bibliotecas front-end empacotadas localmente:
  - `demo/js/jquery-1.8.3.min.js`
  - `demo/js/pica.js`
  - possivelmente outras utilidades JS para manipulação de imagem e UI

### Dependências de build ou pacote

- `bower.json` declara `main: build/convnet.js`, e suporte para:
  - AMD
  - ES6
  - globals
  - Node
  - YUI

## Testes

- Existe um conjunto de testes em `test/jasmine/`
- Arquivos relevantes:
  - `test/jasmine/SpecRunner.html`
  - `test/jasmine/spec/NeuralNetSpec.js`
- O framework de teste usado é **Jasmine 2.0**
- Os testes estão configurados para rodar no browser via HTML runner.

## Informações relevantes adicionais

- O README diz que o projeto não está mais ativamente mantido.
- A compilação do JS é feita por concatenação dos arquivos `src/` e minificação com `yuicompressor`.
- Há suporte histórico para uso com `npm install convnetjs`, mesmo que aqui não exista `package.json`.
- O foco é em pesquisa/educação de redes neurais, não em aplicação web backend.
