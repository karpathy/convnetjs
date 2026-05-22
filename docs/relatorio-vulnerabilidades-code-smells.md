# Relatorio de Vulnerabilidades e Code Smells

## 1. Visao Geral

Este documento apresenta uma triagem estatica local de vulnerabilidades potenciais e code smells identificados no repositorio ConvNetJS. A analise foi baseada nos documentos de requisitos, historias de usuario, casos de uso, plano de refatoracao, README, configuracoes DevSecOps e inspecao dos arquivos do projeto.

Importante: este relatorio nao substitui uma execucao completa de CodeQL, SCA ou pentest. Nao ha arquivo SARIF versionado no repositorio com achados confirmados. Portanto, os itens abaixo representam riscos e smells observados por padroes de codigo, dependencias empacotadas e configuracao do projeto.

## 2. Fontes Analisadas

| Fonte | Finalidade |
| --- | --- |
| `docs/documento-de-requisitos-de-software.md` | Confirmar escopo, requisitos e restricoes do produto. |
| `docs/historias-de-usuario.md` | Relacionar riscos aos perfis de usuario e objetivos do projeto. |
| `docs/casos-de-uso.md` | Mapear impacto dos achados nos fluxos principais. |
| `docs/plano-de-refatoracao.md` | Aproveitar diagnostico tecnico e prioridades de melhoria. |
| `Readme.md` e `sobre este projeto.md` | Confirmar arquitetura, demos, build e manutencao historica. |
| `src/` | Identificar riscos no nucleo da biblioteca. |
| `demo/` | Identificar riscos em exemplos interativos no navegador. |
| `test/` | Avaliar riscos e smells no ambiente de teste. |
| `.github/` | Avaliar pipeline DevSecOps e script de relatorio SARIF. |
| `compile/` e `build/` | Avaliar build legado e artefatos distribuidos. |

## 3. Sumario Executivo

| Categoria | Tipos encontrados | Ocorrencias observadas | Severidade predominante |
| --- | ---: | ---: | --- |
| Vulnerabilidades potenciais | 7 | 103 | Media |
| Code smells | 7 | 2.085 | Media |
| Total | 14 | 2.188 | Media |

Principais pontos:

- O maior risco pratico esta nas demos que usam `eval()` e manipulacao dinamica de HTML.
- O projeto empacota dependencias antigas, como jQuery 1.8.3, Jasmine 2.0.0 e YUI Compressor 2.4.8.
- Existem muitos sinais de codigo legado: `var`, comparacoes nao estritas, dependencia de namespace global e aleatoriedade sem controle.
- A pipeline CodeQL existe, mas exclui `demo/`, `build/`, `test/` e `compile/`, justamente onde parte relevante dos riscos de demo e dependencia aparece.

## 4. Vulnerabilidades Potenciais por Tipo

| ID | Tipo | Quantidade | Severidade | Descricao |
| --- | --- | ---: | --- | --- |
| VULN-01 | Execucao dinamica de codigo com `eval()` | 12 | Alta | Demos permitem executar conteudo de textareas para recriar redes ou agentes. Isso e util para experimentacao local, mas cria risco de XSS ou execucao arbitraria caso a pagina seja publicada, receba entradas nao confiaveis ou seja usada em ambiente compartilhado. |
| VULN-02 | Manipulacao dinamica de HTML | 32 | Media | Uso de `innerHTML` e `jQuery.html()` para renderizar strings. Em varios pontos os valores parecem internos ou gerados pela propria demo, mas o padrao aumenta risco se dados externos ou texto do usuario forem incorporados sem sanitizacao. |
| VULN-03 | Links e recursos HTTP sem TLS | 40 | Media | README, `bower.json` e demos referenciam URLs `http://`. Isso pode permitir alteracao de conteudo em transito, downgrade de seguranca ou bloqueio por navegadores modernos quando usado em contexto HTTPS. |
| VULN-04 | Dependencia legada jQuery 1.8.3 | 8 referencias | Alta | A biblioteca `demo/js/jquery-1.8.3.min.js` e antiga e conhecida por historico de vulnerabilidades em versoes legadas. Mesmo sendo usada nas demos, deve ser atualizada ou isolada de ambientes publicados. |
| VULN-05 | Dependencia legada Jasmine 2.0.0 | 5 referencias | Media | O runner de testes usa Jasmine 2.0.0 empacotado localmente. O risco direto em producao e baixo, mas a dependencia antiga dificulta manutencao e pode carregar padroes inseguros em ambientes de teste publicados. |
| VULN-06 | Ferramenta de build legada YUI Compressor 2.4.8 | 2 referencias | Media | O build depende de um `.jar` antigo em `compile/`. Alem de risco de cadeia de suprimentos e manutencao, dificulta reproducibilidade e auditoria moderna do processo de build. |
| VULN-07 | Script de relatorio com chamada externa condicionada a chave | 4 ocorrencias | Baixa | `.github/scripts/security-review.js` pode chamar a API da OpenAI quando `OPENAI_API_KEY` esta configurada. O fluxo usa variavel de ambiente, mas envia trechos de achados SARIF para servico externo; isso deve ser documentado e tratado como decisao de privacidade/compliance. |

## 5. Detalhamento das Vulnerabilidades

### VULN-01 - Execucao dinamica de codigo com `eval()`

Quantidade: 12 ocorrencias.

Arquivos representativos:

- `demo/js/classify2d.js`
- `demo/js/regression.js`
- `demo/js/image_regression.js`
- `demo/js/images-demo.js`
- `demo/js/autoencoder.js`
- `demo/js/trainers.js`
- `demo/js/rldemo.js`
- `test/jasmine/lib/jasmine-2.0.0/jasmine.js`

Impacto:

- Permite que codigo digitado em campos da pagina seja executado no navegador.
- Pode ser aceitavel em demos locais educacionais, mas e perigoso se hospedado publicamente sem isolamento.
- Afeta principalmente UC-06 e UC-09, pois ambos envolvem demos no navegador.

Recomendacao:

- Substituir `eval()` por parser restrito de configuracao JSON sempre que possivel.
- Quando a execucao dinamica for mantida por finalidade educacional, isolar em sandbox, documentar o risco e evitar hospedar com dados sensiveis.
- Excluir ou proteger demos com `eval()` em publicacoes voltadas a usuarios finais.

### VULN-02 - Manipulacao dinamica de HTML

Quantidade: 32 ocorrencias.

Arquivos representativos:

- `demo/speedtest.html`
- `demo/js/classify2d.js`
- `demo/js/image_regression.js`
- `demo/js/images-demo.js`
- `demo/js/autoencoder.js`
- `demo/js/automatic.js`

Impacto:

- Uso de strings HTML montadas dinamicamente amplia superficie de XSS.
- Parte do conteudo e interno, mas o padrao e arriscado se evoluir para receber entrada de usuario, datasets externos ou parametros de URL.

Recomendacao:

- Preferir `textContent` para texto.
- Criar elementos DOM de forma programatica quando houver dados variaveis.
- Sanitizar qualquer HTML gerado a partir de entrada externa.

### VULN-03 - Links e recursos HTTP sem TLS

Quantidade: 40 ocorrencias.

Arquivos representativos:

- `Readme.md`
- `bower.json`
- `demo/*.html`
- `demo/js/npgmain.js`
- `demo/js/pica.js`
- `test/jasmine/lib/jasmine-2.0.0/*.js`

Impacto:

- Recursos carregados via HTTP podem ser alterados em transito.
- Navegadores modernos podem bloquear conteudo misto se as demos forem servidas por HTTPS.
- Documentacao aponta usuarios para links antigos e potencialmente inseguros.

Recomendacao:

- Migrar links externos para `https://` quando os destinos suportarem TLS.
- Evitar carregar fontes ou scripts remotos por HTTP.
- Para links historicos sem HTTPS, manter como referencia textual e nao como recurso carregado automaticamente.

### VULN-04 - Dependencia legada jQuery 1.8.3

Quantidade: 8 referencias ao arquivo nas demos.

Arquivos representativos:

- `demo/js/jquery-1.8.3.min.js`
- `demo/automatic.html`
- `demo/autoencoder.html`
- `demo/cifar10.html`
- `demo/classify2d.html`
- `demo/image_regression.html`
- `demo/mnist.html`
- `demo/trainers.html`

Impacto:

- jQuery 1.8.3 e muito antigo para padroes atuais.
- Pode conter vulnerabilidades conhecidas e APIs inseguras, especialmente quando combinado com `.html()` e entradas dinamicas.

Recomendacao:

- Atualizar jQuery para versao mantida ou remover dependencia onde for simples.
- Testar todas as demos apos atualizacao.
- Se a atualizacao quebrar demos historicas, documentar risco e limitar uso a ambiente local.

### VULN-05 - Dependencia legada Jasmine 2.0.0

Quantidade: 5 referencias diretas no runner.

Arquivos representativos:

- `test/jasmine/SpecRunner.html`
- `test/jasmine/lib/jasmine-2.0.0/`

Impacto:

- Baixo impacto em producao, pois fica em `test/`.
- Mantem stack de testes antiga, dificultando automacao moderna e auditoria.

Recomendacao:

- Planejar atualizacao do runner de testes.
- Manter compatibilidade com testes browser-only enquanto uma alternativa moderna nao for validada.

### VULN-06 - YUI Compressor 2.4.8

Quantidade: 2 referencias.

Arquivos representativos:

- `compile/yuicompressor-2.4.8.jar`
- `compile/build.xml`
- `Readme.md`

Impacto:

- Ferramenta antiga e empacotada como binario `.jar`.
- Dificulta verificacao de procedencia e modernizacao da cadeia de build.

Recomendacao:

- Manter o build atual ate existir cobertura de regressao.
- Avaliar substituto moderno de minificacao em paralelo.
- Documentar hash, origem e uso do `.jar` enquanto ele permanecer no repositorio.

### VULN-07 - Envio opcional de SARIF para servico externo

Quantidade: 4 ocorrencias relacionadas.

Arquivos representativos:

- `.github/scripts/security-review.js`
- `.github/workflows/devsecops-security.yml`

Impacto:

- Quando `OPENAI_API_KEY` esta configurada, achados SARIF sao enviados para analise externa.
- Pode haver exposicao de nomes de arquivos, mensagens de regras e trechos de contexto.

Recomendacao:

- Documentar explicitamente esse comportamento no workflow.
- Permitir desativacao por variavel de ambiente.
- Revisar politica de dados antes de usar em repositorios privados ou sensiveis.

## 6. Code Smells por Tipo

| ID | Tipo | Quantidade | Severidade | Descricao |
| --- | --- | ---: | --- | --- |
| SMELL-01 | Uso extensivo de `var` | 1.610 | Media | Indica estilo JavaScript legado, escopo de funcao e maior risco de hoisting inesperado. |
| SMELL-02 | Comparacoes nao estritas | 293 | Media | Uso potencial de `==` ou `!=` aumenta risco de coercao implicita e comportamento inesperado. |
| SMELL-03 | Acoplamento ao namespace global | 110 | Media | Uso frequente de `global.*` e registro em `convnetjs` cria acoplamento por ordem de carregamento. |
| SMELL-04 | Aleatoriedade nao controlada | 34 | Media | Uso de `Math.random` em codigo e testes torna resultados menos reproduziveis. |
| SMELL-05 | Logs diretos no codigo | 17 | Baixa | `console.log` aparece em biblioteca, demos, testes e scripts; dificulta controle de saida e tratamento padronizado de erro. |
| SMELL-06 | Comentarios TODO/FIXME | 14 | Baixa | Indicam dividas tecnicas conhecidas, incluindo perda e organizacao de utilitarios. |
| SMELL-07 | Build e testes fortemente legados | 7 exports/referencias criticas | Media | Uso de IIFEs, `module.exports`, `window.convnetjs`, Ant e runner browser-only exige cuidado com ordem de arquivos e dificulta CI moderna. |

## 7. Detalhamento dos Code Smells

### SMELL-01 - Uso extensivo de `var`

Quantidade: 1.610 ocorrencias.

Impacto:

- Escopo de funcao pode causar bugs sutis em loops e callbacks.
- Dificulta aplicacao de regras modernas de lint.
- Reflete padrao historico do projeto, mas aumenta custo de manutencao.

Recomendacao:

- Migrar gradualmente para `let` e `const`, com testes antes e depois.
- Priorizar `src/` antes de `demo/` e dependencias empacotadas.

### SMELL-02 - Comparacoes nao estritas

Quantidade: 293 ocorrencias.

Impacto:

- Coercao implicita pode mascarar valores inesperados.
- Em codigo numerico, diferencas entre string, numero e boolean podem gerar comportamentos dificeis de diagnosticar.

Recomendacao:

- Migrar para `===` e `!==` onde a equivalencia estrita preserve o comportamento.
- Usar testes de regressao para evitar mudancas semanticas.

### SMELL-03 - Acoplamento ao namespace global

Quantidade: 110 ocorrencias de `global.*` em `src/` e `build/`.

Impacto:

- A ordem de concatenacao em `compile/build.xml` se torna critica.
- Refatoracoes em arquivos isolados podem quebrar outros modulos silenciosamente.
- Dificulta uso de ferramentas modernas de analise e bundling.

Recomendacao:

- Introduzir fabrica interna de camadas e mapa central de exports.
- Preservar `window.convnetjs` e `module.exports` como API publica.
- Refatorar internamente antes de alterar formato de modulo.

### SMELL-04 - Aleatoriedade nao controlada

Quantidade: 34 ocorrencias.

Impacto:

- Testes podem falhar ou passar dependendo da sorte.
- Experimentos nao sao plenamente reproduziveis.
- Afeta inicializacao de pesos, dropout, MagicNet, demos e testes.

Recomendacao:

- Criar gerador de numeros aleatorios injetavel ou configuravel para testes.
- Manter `Math.random` como padrao para compatibilidade quando nenhuma semente for definida.

### SMELL-05 - Logs diretos no codigo

Quantidade: 17 ocorrencias.

Impacto:

- Alguns erros sao apenas impressos, sem interromper execucao.
- Dificulta automacao de testes que esperam excecoes ou resultados estruturados.
- Mistura diagnostico, aviso e erro em `console.log`.

Recomendacao:

- Padronizar erros e warnings.
- Substituir logs criticos por excecoes testaveis quando isso nao quebrar demos.
- Manter logs de demo apenas quando forem parte da visualizacao educacional.

### SMELL-06 - TODO/FIXME

Quantidade: 14 ocorrencias.

Impacto:

- Indica pontos conhecidos de divida tecnica.
- Alguns TODOs aparecem em dependencias de teste empacotadas, mas ha tambem comentario relevante em `src/convnet_trainers.js` sobre reorganizacao de calculo de loss.

Recomendacao:

- Classificar TODOs entre codigo proprio e dependencias empacotadas.
- Converter TODOs do codigo proprio em itens de backlog rastreaveis.

### SMELL-07 - Build e testes legados

Quantidade: 7 referencias criticas a exports browser/Node e processo antigo.

Impacto:

- Build por concatenacao depende de ordem fixa.
- Testes dependem de runner HTML manual.
- Modernizacao direta pode quebrar API publica e demos.

Recomendacao:

- Preservar fluxo atual ate ampliar cobertura de testes.
- Adicionar comandos automatizaveis para smoke tests.
- Avaliar build moderno somente em paralelo, conforme definido no plano de refatoracao.

## 8. Relacao com Casos de Uso

| Caso de uso | Riscos mais relevantes | Observacao |
| --- | --- | --- |
| UC-01 - Definir rede neural | SMELL-03, SMELL-05, SMELL-07 | Acoplamento global e validacoes por log dificultam evolucao segura de `makeLayers`. |
| UC-02 - Treinar modelo | SMELL-04, SMELL-05 | Aleatoriedade e logs dificultam reprodutibilidade e diagnostico de treino. |
| UC-03 - Executar inferencia | SMELL-03, SMELL-04 | API publica deve ser preservada durante qualquer modernizacao. |
| UC-04 - Usar CNN com imagem | VULN-02, SMELL-04 | Demos de imagem manipulam DOM e dependem de dados dinamicos. |
| UC-05 - Serializar e restaurar rede | SMELL-03, SMELL-07 | Mudancas internas podem quebrar modelos serializados se nao houver testes. |
| UC-06 - Executar demo no navegador | VULN-01, VULN-02, VULN-03, VULN-04 | Principal concentracao de riscos de seguranca esta nas demos. |
| UC-07 - Compilar biblioteca | VULN-06, SMELL-07 | Build legado deve ser estabilizado antes de modernizacao. |
| UC-08 - Rodar testes Jasmine | VULN-05, SMELL-07 | Testes antigos precisam ser preservados enquanto se planeja atualizacao. |
| UC-09 - Experimentar Deep Q Learning | VULN-01, SMELL-04 | Demo de RL usa configuracao dinamica e comportamento experimental. |

## 9. Priorizacao de Correcao

| Prioridade | Itens | Acao recomendada |
| --- | --- | --- |
| Alta | VULN-01, VULN-04 | Reduzir ou isolar `eval()` nas demos e atualizar/remover jQuery legado. |
| Alta | SMELL-03, SMELL-07 | Mapear API publica e proteger ordem de build antes de refatorar. |
| Media | VULN-02, VULN-03, VULN-06 | Sanitizar HTML dinamico, migrar links para HTTPS e documentar/substituir YUI Compressor. |
| Media | SMELL-01, SMELL-02, SMELL-04 | Migrar JS legado gradualmente, usar comparacoes estritas e controlar aleatoriedade. |
| Baixa | VULN-05, VULN-07, SMELL-05, SMELL-06 | Atualizar Jasmine, documentar envio externo SARIF, padronizar logs e converter TODOs em backlog. |

## 10. Plano de Acao Resumido

1. Executar CodeQL e anexar o SARIF real ao processo de revisao.
2. Revisar demos com `eval()` e decidir entre remover, substituir por JSON ou isolar por sandbox.
3. Atualizar jQuery ou limitar demos antigas a ambiente local documentado.
4. Migrar links HTTP para HTTPS.
5. Criar testes de regressao para `Net`, `Vol`, `Trainer`, serializacao e demos principais.
6. Padronizar tratamento de erros antes de grandes refatoracoes.
7. Modernizar build somente apos cobertura minima de regressao.

## 11. Limitacoes da Analise

- As quantidades foram obtidas por busca textual estatica e podem incluir ocorrencias em dependencias empacotadas.
- Nao foi executada auditoria online de CVEs ou banco de vulnerabilidades.
- Nao foi executado CodeQL local com resultados SARIF neste relatorio.
- Nem toda ocorrencia representa vulnerabilidade exploravel; alguns achados sao riscos condicionais ao modo como as demos forem publicadas ou usadas.

