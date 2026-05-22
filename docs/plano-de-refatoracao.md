# Plano de Refatoracao do ConvNetJS

## 1. Visao Geral

Este plano define uma estrategia de refatoracao incremental para o ConvNetJS, com base nos documentos de requisitos, historias de usuario, casos de uso, README e estrutura tecnica do repositorio.

O objetivo nao e reescrever a biblioteca nem alterar seu comportamento publico. A refatoracao deve preservar a API existente, os demos estaticos e a compatibilidade historica com navegador e Node.js, enquanto melhora testabilidade, manutenibilidade, organizacao interna e seguranca de evolucao.

## 2. Objetivos da Refatoracao

| ID | Objetivo | Relacao com requisitos |
| --- | --- | --- |
| OR-01 | Reduzir risco de alteracoes no motor de redes neurais por meio de testes mais amplos e repetiveis. | RF-01 a RF-14, RF-20, RNF-07 |
| OR-02 | Preservar compatibilidade da API publica `convnetjs`, `Net`, `Vol`, `Trainer` e `SGDTrainer`. | RF-16, RF-17, HU-06 |
| OR-03 | Melhorar validacoes e mensagens de erro sem quebrar fluxos existentes. | RF-01, RF-02, REG-01, REG-02 |
| OR-04 | Isolar responsabilidades internas de rede, camadas, treinadores, volumes e utilitarios. | RNF-03 |
| OR-05 | Modernizar gradualmente o processo de build e teste, mantendo os artefatos em `build/`. | RF-19, RF-20, RNF-04 |
| OR-06 | Manter demos como exemplos estaticos funcionais e nao transforma-las em aplicacao web. | RF-18, RNF-05 |

## 3. Principios e Restricoes

- Preservar comportamento antes de melhorar desenho interno.
- Refatorar em etapas pequenas, sempre acompanhadas por testes.
- Manter os nomes publicos usados por demos e exemplos: `convnetjs.Net`, `convnetjs.Vol`, `convnetjs.Trainer`, `convnetjs.SGDTrainer`, `convnetjs.img_to_vol` e demais exports existentes.
- Nao introduzir backend, banco de dados, autenticacao, APIs externas obrigatorias ou fluxo SaaS.
- Nao remover `build/convnet.js` e `build/convnet-min.js` sem substituir o processo de distribuicao de forma compativel.
- Tratar o modulo de Deep Q Learning como experimental e separado do nucleo principal.
- Priorizar regressao funcional de `src/` antes de alterar demos ou build.

## 4. Diagnostico Tecnico

| Area | Situacao atual | Risco |
| --- | --- | --- |
| Modularizacao | Arquivos em `src/` usam IIFEs e registram simbolos em `convnetjs`. | Mudancas podem quebrar ordem de carregamento e exports globais. |
| API publica | API historica e usada diretamente por demos e README. | Renomear ou mover simbolos quebra usuarios existentes. |
| Build | `compile/build.xml` concatena arquivos em ordem fixa e minifica com YUI Compressor. | Processo antigo, sensivel a ordem dos arquivos e dificil de integrar a CI moderna. |
| Testes | Jasmine 2.0 via `SpecRunner.html`, com cobertura basica de rede densa. | Refatoracoes em camadas, serializacao e otimizadores podem passar sem cobertura. |
| Aleatoriedade | Uso amplo de `Math.random` em utilitarios, inicializacao, dropout, MagicNet e demos. | Testes podem ser instaveis e resultados dificeis de reproduzir. |
| Validacao | Alguns erros usam `assert`, outros usam `console.log`. | Falhas podem nao interromper fluxo ou nao serem faceis de testar. |
| Demos | Paginas HTML/JS estaticas acopladas ao bundle e scripts locais. | Alteracoes de API ou caminhos podem quebrar exemplos educacionais. |
| Serializacao | `toJSON` e `fromJSON` existem em rede, volumes e camadas. | Mudancas internas podem quebrar modelos salvos. |

## 5. Roadmap de Refatoracao

### Fase 0 - Baseline e Inventario

Prioridade: Alta

Atividades:

- Registrar a API publica atual exportada por `src/convnet_export.js` e pelo bundle `build/convnet.js`.
- Criar uma lista de demos que devem continuar carregando: MNIST, CIFAR-10, classificacao 2D, regressao, autoencoder, trainers, image regression e RL.
- Documentar a ordem atual de concatenacao do `compile/build.xml` como contrato temporario.
- Confirmar quais arquivos em `build/` sao gerados e quais sao fontes mantidas manualmente.

Criterios de aceite:

- Existe uma lista de simbolos publicos preservados.
- Existe uma lista de demos criticas para smoke test.
- O time sabe quais artefatos devem ser regenerados apos mudancas em `src/`.

### Fase 1 - Fortalecimento de Testes

Prioridade: Alta

Atividades:

- Expandir `test/jasmine/spec/NeuralNetSpec.js` ou criar novos specs para cobrir:
  - criacao de redes com `fc`, `conv`, `pool`, ativacoes, dropout e normalizacao;
  - `forward`, `backward`, `getPrediction`, `getCostLoss`;
  - serializacao e restauracao com `toJSON` e `fromJSON`;
  - treinadores `sgd`, `adam`, `adagrad`, `adadelta`, `windowgrad` e `nesterov`;
  - utilitarios de `Vol`, incluindo `clone`, `cloneAndZero`, `addFrom` e gradientes.
- Introduzir testes deterministas para cenarios que hoje dependem de aleatoriedade.
- Criar um smoke test documentado para abrir `SpecRunner.html` e demos principais.

Criterios de aceite:

- Os casos UC-01 a UC-05 possuem cobertura automatizada minima.
- Os testes conseguem detectar quebra em criacao de rede, treino, inferencia e serializacao.
- Testes instaveis por aleatoriedade sao reduzidos ou isolados.

### Fase 2 - Validacoes e Tratamento de Erros

Prioridade: Alta

Atividades:

- Padronizar validacoes internas para configuracoes invalidas de rede, camadas e treinadores.
- Substituir mensagens criticas via `console.log` por erros testaveis onde isso nao quebrar compatibilidade esperada.
- Preservar mensagens ou aliases quando forem potencialmente observados por usuarios.
- Validar explicitamente tipos de camada desconhecidos em `Net.makeLayers`.
- Validar formato minimo de JSON em `fromJSON` para rede e camadas.

Criterios de aceite:

- REG-01, REG-02 e REG-03 sao verificaveis por testes.
- Configuracoes invalidas falham de forma previsivel.
- Demos e exemplos existentes continuam funcionando sem alteracoes obrigatorias de uso.

### Fase 3 - Separacao Interna de Responsabilidades

Prioridade: Media

Atividades:

- Extrair a logica de "desugar" de camadas de `Net.makeLayers` para uma funcao interna testavel.
- Centralizar o mapeamento entre `type` de camada e construtor em uma fabrica interna.
- Reduzir duplicacao entre `makeLayers` e `fromJSON` para criacao de camadas.
- Organizar utilitarios de erro, validacao e opcoes sem alterar o namespace publico.
- Manter os arquivos atuais em `src/` durante esta fase para preservar o build por concatenacao.

Criterios de aceite:

- `Net.makeLayers` fica menor e mais legivel.
- A criacao de camadas por definicao e por JSON usa uma fonte comum de mapeamento.
- Nenhum nome publico e removido.

### Fase 4 - Refatoracao de Treinadores e Aleatoriedade

Prioridade: Media

Atividades:

- Separar calculo de atualizacao dos otimizadores em funcoes internas por metodo.
- Manter `Trainer` e `SGDTrainer` como aliases publicos compativeis.
- Introduzir um ponto interno para geracao de numeros aleatorios, com possibilidade de semente em testes.
- Aplicar determinismo primeiro nos testes, depois avaliar exposicao opcional na API.
- Revisar calculo de perdas de regularizacao para facilitar testes de regressao.

Criterios de aceite:

- Cada metodo de otimizacao pode ser testado isoladamente.
- Os testes de treinamento deixam de depender fortemente de sorte estatistica.
- O comportamento publico de `trainer.train(x, y)` e mantido.

### Fase 5 - Build, Distribuicao e Compatibilidade

Prioridade: Media

Atividades:

- Manter o build Ant existente inicialmente como caminho oficial.
- Criar um script auxiliar moderno somente se ele gerar resultado funcionalmente equivalente ao bundle atual.
- Documentar claramente como regenerar `build/convnet.js` e `build/convnet-min.js`.
- Adicionar verificacao de que o bundle exporta `window.convnetjs` no navegador e `module.exports` no Node.js.
- Evitar troca completa para TypeScript, ESM ou bundler moderno antes de estabilizar testes.

Criterios de aceite:

- RF-16, RF-17 e RF-19 continuam atendidos.
- O bundle gerado passa nos mesmos testes do codigo fonte.
- O README ou documento tecnico informa o processo recomendado de build.

### Fase 6 - Demos e Documentacao Tecnica

Prioridade: Media

Atividades:

- Revisar demos para remover dependencias desnecessarias de caminhos quebradicos, sem redesenhar a UI.
- Adicionar notas curtas nas demos ou documentacao sobre seu carater estatico e educacional.
- Garantir que exemplos do README continuem validos apos refatoracoes.
- Documentar limitacoes conhecidas: manutencao historica limitada, ausencia de GPU, ausencia de backend e modulo RL experimental.

Criterios de aceite:

- UC-06 e UC-09 continuam funcionando como demos estaticas.
- Exemplos basicos de criacao, treino e inferencia permanecem corretos.
- A documentacao nao promete capacidades fora do escopo do projeto.

## 6. Backlog Priorizado

| Prioridade | Item | Resultado esperado |
| --- | --- | --- |
| Alta | Mapear API publica atual | Evita que refatoracao quebre usuarios existentes. |
| Alta | Ampliar testes de `Net`, `Vol`, camadas e serializacao | Cria rede de seguranca para mudancas internas. |
| Alta | Cobrir `Trainer` e otimizadores com testes deterministas | Reduz risco em RF-10 e RF-11. |
| Alta | Padronizar falhas de configuracao em `makeLayers` | Melhora previsibilidade de RF-01 e RF-02. |
| Media | Extrair fabrica de camadas | Reduz duplicacao entre criacao e restauracao. |
| Media | Extrair `desugar` de camadas | Facilita manutencao de ativacoes, dropout e perdas. |
| Media | Criar controle interno de aleatoriedade para testes | Reduz flakiness e facilita reproducibilidade. |
| Media | Documentar build atual e smoke tests de demos | Protege RF-18 e RF-19. |
| Baixa | Isolar Deep Q Learning como modulo experimental documentado | Mantem escopo claro de RF-21. |
| Baixa | Avaliar build moderno em paralelo ao Ant | Possivel melhoria futura sem quebra imediata. |

## 7. Estrategia de Testes

| Tipo de teste | Escopo | Relacao |
| --- | --- | --- |
| Unitario | `Vol`, utilitarios, fabrica de camadas, validacoes e otimizadores. | RF-01 a RF-14 |
| Integracao | Criar rede, treinar, inferir, serializar e restaurar. | UC-01 a UC-05 |
| Regressao de bundle | Carregar `build/convnet.js` em browser e Node.js. | RF-16, RF-17 |
| Smoke de demos | Abrir demos principais e verificar inicializacao sem erro. | UC-06, UC-09 |
| Build | Executar processo de compilacao e verificar artefatos gerados. | UC-07 |

## 8. Riscos e Mitigacoes

| Risco | Impacto | Mitigacao |
| --- | --- | --- |
| Quebra da API publica historica | Alto | Mapear exports antes da refatoracao e testar aliases. |
| Mudanca na ordem de carregamento dos arquivos | Alto | Preservar `build.xml` ate haver substituto validado. |
| Testes instaveis por `Math.random` | Medio | Introduzir controle de aleatoriedade nos testes. |
| Regressao em demos antigas | Medio | Criar checklist de smoke test para paginas principais. |
| Incompatibilidade com modelos serializados | Alto | Criar testes de `toJSON`/`fromJSON` antes de alterar camadas. |
| Refatoracao grande demais em uma etapa | Alto | Executar fases pequenas com revisao e teste a cada etapa. |
| Modernizacao prematura de build | Medio | Tratar build moderno como opcional ate preservar saida equivalente. |

## 9. Mapeamento com Casos de Uso

| Caso de uso | Impacto da refatoracao | Fases relacionadas |
| --- | --- | --- |
| UC-01 - Definir rede neural | Melhorar validacao, fabrica de camadas e legibilidade de `makeLayers`. | Fases 1, 2, 3 |
| UC-02 - Treinar modelo | Isolar otimizadores e testar atualizacao de parametros. | Fases 1, 4 |
| UC-03 - Executar inferencia | Preservar `forward`, `getPrediction` e contrato de `Vol`. | Fases 1, 2 |
| UC-04 - Usar CNN com imagem | Garantir cobertura de `conv`, `pool` e conversao de imagem. | Fases 1, 6 |
| UC-05 - Serializar e restaurar rede | Proteger compatibilidade de JSON. | Fases 1, 2, 3 |
| UC-06 - Executar demo no navegador | Validar demos apos mudancas em API e build. | Fases 5, 6 |
| UC-07 - Compilar biblioteca | Documentar e estabilizar build. | Fase 5 |
| UC-08 - Rodar testes Jasmine | Ampliar cobertura e confiabilidade. | Fase 1 |
| UC-09 - Experimentar Deep Q Learning | Manter modulo experimental isolado. | Fase 6 |

## 10. Criterios de Conclusao

A refatoracao deve ser considerada concluida quando:

- A API publica essencial permanece compativel com README, demos e casos de uso.
- Os testes cobrem criacao de rede, treinamento, inferencia, serializacao, camadas principais e otimizadores.
- O bundle distribuivel continua sendo gerado e carregado em browser e Node.js.
- As demos principais continuam funcionando como paginas estaticas.
- O codigo de `Net`, treinadores e criacao de camadas fica mais coeso e testavel.
- A documentacao tecnica deixa claro como testar, compilar e validar regressao.

## 11. Ordem Recomendada de Execucao

1. Criar inventario da API publica e checklist de demos.
2. Ampliar testes antes de alterar codigo produtivo.
3. Padronizar validacoes e erros de configuracao.
4. Refatorar criacao de camadas em `Net`.
5. Refatorar treinadores e aleatoriedade com testes deterministas.
6. Validar build e exports browser/Node.js.
7. Revisar demos e documentacao apos estabilizacao.

