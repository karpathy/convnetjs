# Documento de Requisitos de Software

## 1. Identificacao do Projeto

| Campo | Descricao |
| --- | --- |
| Projeto | ConvNetJS |
| Tipo de software | Biblioteca JavaScript para redes neurais e redes convolucionais |
| Linguagem principal | JavaScript |
| Ambientes de execucao | Navegador e Node.js |
| Artefato principal | `build/convnet.js` |
| Fonte principal de requisitos | `sobre este projeto.md`, `Readme.md`, `src/`, `demo/` e `test/jasmine/` |

## 2. Visao Geral

ConvNetJS e uma biblioteca JavaScript para criacao, treinamento e execucao de redes neurais, com suporte a redes totalmente conectadas, redes convolucionais, funcoes de perda, otimizadores e demos educacionais executadas no navegador.

O projeto nao e uma aplicacao web com backend, banco de dados, autenticacao ou area administrativa. Seu objetivo principal e oferecer um motor de Deep Learning em JavaScript, utilizavel por desenvolvedores, estudantes e pesquisadores em contextos educacionais, experimentais e de prototipacao.

## 3. Proposito

Este documento registra os requisitos de software levantados a partir do estado atual do repositorio. Ele serve como referencia para entendimento do escopo, validacao das funcionalidades existentes, manutencao futura e comunicacao entre pessoas tecnicas e nao tecnicas envolvidas no projeto.

## 4. Escopo

### 4.1 Dentro do escopo

- Criacao de redes neurais por definicoes de camadas.
- Execucao de propagacao direta para inferencia.
- Treinamento por retropropagacao usando treinadores disponiveis.
- Suporte a camadas densas, convolucionais, pooling, dropout, normalizacao e ativacoes.
- Suporte a tarefas de classificacao, regressao e modulo experimental de aprendizado por reforco.
- Serializacao e restauracao de redes via JSON.
- Uso em navegador e Node.js/CommonJS.
- Demos estaticas para aprendizado e experimentacao.
- Testes de comportamento basico com Jasmine.

### 4.2 Fora do escopo

- Cadastro, login, controle de acesso ou gestao de usuarios.
- Banco de dados, tabelas, migracoes ou persistencia em servidor.
- Backend proprio, APIs REST ou integracoes externas obrigatorias.
- Treinamento distribuido, uso de GPU ou execucao de alto desempenho em producao.
- Interface administrativa, dashboard SaaS ou produto hospedado.

## 5. Publico-Alvo

| Perfil | Interesse principal |
| --- | --- |
| Desenvolvedor JavaScript | Integrar a biblioteca em projetos browser ou Node.js |
| Estudante/Pesquisador | Aprender e experimentar conceitos de redes neurais |
| Usuario de demo | Visualizar exemplos interativos no navegador |
| Mantenedor | Corrigir, testar, compilar e distribuir a biblioteca |
| Integrador | Usar o bundle compilado ou a API CommonJS em outro sistema |

## 6. Visao do Produto

O ConvNetJS disponibiliza uma API programatica baseada em objetos como `Net`, `Vol` e `SGDTrainer`. O usuario define uma lista de camadas, cria a rede, executa inferencia com `forward`, treina a rede com um treinador e pode salvar ou restaurar o estado por JSON.

A biblioteca tambem inclui demos HTML/JS que demonstram classificacao, regressao, autoencoder, CNN para imagens, comparacao de otimizadores e aprendizado por reforco com Deep Q Learning.

## 7. Requisitos Funcionais

| ID | Requisito | Prioridade | Origem |
| --- | --- | --- | --- |
| RF-01 | O sistema deve permitir definir redes neurais a partir de uma lista ordenada de camadas. | Alta | `src/convnet_net.js` |
| RF-02 | O sistema deve validar que a primeira camada da rede seja uma camada de entrada. | Alta | `src/convnet_net.js` |
| RF-03 | O sistema deve suportar camadas totalmente conectadas para processamento neural denso. | Alta | `src/convnet_layers_dotproducts.js` |
| RF-04 | O sistema deve suportar camadas convolucionais para processamento de dados de imagem ou volumes. | Alta | `src/convnet_layers_dotproducts.js` |
| RF-05 | O sistema deve suportar camadas de pooling para reducao espacial de volumes. | Alta | `src/convnet_layers_pool.js` |
| RF-06 | O sistema deve suportar funcoes de ativacao como ReLU, sigmoid, tanh e maxout. | Alta | `src/convnet_layers_nonlinearities.js` |
| RF-07 | O sistema deve suportar dropout e normalizacao local quando configurados na rede. | Media | `src/convnet_layers_dropout.js`, `src/convnet_layers_normalization.js` |
| RF-08 | O sistema deve suportar funcoes de perda para classificacao e regressao, incluindo softmax, SVM e L2. | Alta | `src/convnet_layers_loss.js` |
| RF-09 | O sistema deve executar propagacao direta e retornar um volume de saida. | Alta | `src/convnet_net.js` |
| RF-10 | O sistema deve executar retropropagacao para calculo de perdas e gradientes. | Alta | `src/convnet_net.js` |
| RF-11 | O sistema deve fornecer treinadores/otimizadores para ajuste dos parametros da rede. | Alta | `src/convnet_trainers.js` |
| RF-12 | O sistema deve permitir obter a predicao de maior probabilidade em redes com softmax. | Media | `src/convnet_net.js` |
| RF-13 | O sistema deve permitir serializar e restaurar redes por JSON. | Alta | `src/convnet_net.js` |
| RF-14 | O sistema deve fornecer estruturas de volume para armazenar dados, pesos e gradientes. | Alta | `src/convnet_vol.js` |
| RF-15 | O sistema deve disponibilizar utilitario para converter imagem em volume quando usado no navegador. | Media | `Readme.md`, `src/convnet_vol_util.js` |
| RF-16 | O sistema deve expor a biblioteca como objeto global no navegador. | Alta | `src/convnet_export.js` |
| RF-17 | O sistema deve exportar a biblioteca via CommonJS quando executado em Node.js. | Alta | `src/convnet_export.js` |
| RF-18 | O sistema deve disponibilizar demos estaticas para exemplos de classificacao, regressao, CNN, autoencoder, treinamento e aprendizado por reforco. | Media | `demo/`, `Readme.md` |
| RF-19 | O sistema deve disponibilizar script de compilacao para gerar versoes concatenada e minificada da biblioteca. | Media | `compile/build.xml` |
| RF-20 | O sistema deve conter testes automatizados basicos para verificar inicializacao, inferencia, treinamento e gradientes. | Media | `test/jasmine/spec/NeuralNetSpec.js` |
| RF-21 | O sistema deve disponibilizar modulo experimental de Deep Q Learning. | Baixa | `build/deepqlearn.js`, `demo/rldemo.html` |

## 8. Requisitos Nao Funcionais

| ID | Requisito | Prioridade | Categoria |
| --- | --- | --- | --- |
| RNF-01 | A biblioteca deve ser executavel em navegadores por meio de arquivos JavaScript estaticos. | Alta | Portabilidade |
| RNF-02 | A biblioteca deve manter compatibilidade com uso em Node.js/CommonJS conforme exportacao existente. | Alta | Compatibilidade |
| RNF-03 | O codigo-fonte deve permanecer modular em `src/`, separado por responsabilidade de camadas, volumes, rede, treinadores e utilitarios. | Media | Manutenibilidade |
| RNF-04 | A distribuicao deve oferecer arquivo compilado e arquivo minificado em `build/`. | Media | Distribuicao |
| RNF-05 | As demos devem funcionar sem backend proprio, usando HTML, CSS e JavaScript estaticos. | Alta | Implantacao |
| RNF-06 | A biblioteca deve evitar dependencia obrigatoria de APIs externas para seu funcionamento principal. | Alta | Independencia |
| RNF-07 | Os testes devem ser executaveis no navegador por meio do runner Jasmine existente. | Media | Testabilidade |
| RNF-08 | A API deve permitir exemplos curtos e compreensiveis para fins educacionais. | Media | Usabilidade |
| RNF-09 | A execucao deve ser adequada a experimentos pequenos e medios em JavaScript, sem promessa de desempenho de producao com GPU. | Media | Desempenho |
| RNF-10 | O projeto deve respeitar a licenca MIT presente no repositorio. | Alta | Legal |
| RNF-11 | O uso das demos deve ser restrito ao processamento local no navegador, sem envio obrigatorio de dados a servidor. | Media | Privacidade |
| RNF-12 | O projeto deve preservar o aviso de manutencao limitada indicado no README. | Media | Governanca |

## 9. Regras de Negocio e Restricoes

| ID | Regra/Restricao | Descricao |
| --- | --- | --- |
| REG-01 | Camada inicial obrigatoria | Toda rede criada pela API deve iniciar com uma camada do tipo `input`. |
| REG-02 | Rede minima | Uma rede deve possuir ao menos uma camada de entrada e uma camada final de perda ou saida adequada. |
| REG-03 | Saida softmax | A funcao de predicao por maior probabilidade pressupoe que a ultima camada seja `softmax`. |
| REG-04 | Demos estaticas | As demonstracoes devem ser tratadas como paginas estaticas de exemplo, nao como aplicacao web completa. |
| REG-05 | Sem persistencia em servidor | Persistencia, quando necessaria, deve ocorrer por mecanismos locais como serializacao JSON, nao por banco de dados do projeto. |
| REG-06 | Modulo experimental | O modulo de Deep Q Learning deve ser documentado como experimental. |

## 10. Interfaces Externas

| Interface | Descricao |
| --- | --- |
| Browser global | Exposicao da biblioteca como `window.convnetjs`. |
| CommonJS | Exportacao via `module.exports` para uso com `require`. |
| Arquivos HTML de demo | Paginas em `demo/` que consomem scripts locais e a biblioteca compilada. |
| Jasmine SpecRunner | Runner HTML para execucao dos testes automatizados existentes. |
| Build Ant | Script `compile/build.xml` para concatenacao e minificacao. |

## 11. Premissas

- O levantamento descreve o software existente, nao uma nova versao do produto.
- O projeto e usado principalmente para estudo, prototipacao e demonstracao.
- Nao ha banco de dados, backend, autenticacao ou integracao externa obrigatoria.
- A manutencao do projeto e historicamente limitada, conforme aviso do README.
- Os requisitos foram derivados por analise documental e inspecao do repositorio.

## 12. Matriz de Rastreabilidade

| Requisito | Historias relacionadas | Casos de uso relacionados |
| --- | --- | --- |
| RF-01, RF-02 | HU-01 | UC-01 |
| RF-03, RF-04, RF-05, RF-06, RF-07, RF-08 | HU-01, HU-02 | UC-01, UC-04 |
| RF-09, RF-12 | HU-03 | UC-03 |
| RF-10, RF-11 | HU-02 | UC-02 |
| RF-13 | HU-04 | UC-05 |
| RF-14, RF-15 | HU-03, HU-05 | UC-03, UC-04 |
| RF-16, RF-17 | HU-06 | UC-01, UC-03 |
| RF-18 | HU-05, HU-07 | UC-06 |
| RF-19 | HU-08 | UC-07 |
| RF-20 | HU-09 | UC-08 |
| RF-21 | HU-10 | UC-09 |
| RNF-01, RNF-02, RNF-05, RNF-06 | HU-06, HU-07 | UC-06 |
| RNF-03, RNF-04, RNF-07 | HU-08, HU-09 | UC-07, UC-08 |

