# Contabilizacao de Problemas para Refatoracao

## 1. Objetivo

Este documento consolida os problemas identificados no projeto ConvNetJS por tipo, separando achados de seguranca e achados de qualidade de codigo. Ele deve servir como base quantitativa e qualitativa para priorizacao do plano de refatoracao.

A contabilizacao foi derivada principalmente de:

- `docs/plano-de-refatoracao.md`
- `docs/relatorio-vulnerabilidades-code-smells.md`
- `docs/documento-de-requisitos-de-software.md`
- `docs/historias-de-usuario.md`
- `docs/casos-de-uso.md`
- inspecao estatica dos diretorios `src/`, `demo/`, `test/`, `compile/`, `build/` e `.github/`

## 2. Resumo Geral

| Tipo de problema | Quantidade de categorias | Ocorrencias contabilizadas | Severidade predominante | Prioridade de tratamento |
| --- | ---: | ---: | --- | --- |
| Seguranca | 7 | 103 | Media | Alta |
| Qualidade | 7 | 2.085 | Media | Alta |
| Total | 14 | 2.188 | Media | Alta |

Observacao: as quantidades representam ocorrencias encontradas por busca estatica textual. Algumas ocorrencias aparecem em dependencias empacotadas, demos ou arquivos de teste, portanto devem ser tratadas conforme contexto antes de qualquer correcao automatica.

## 3. Problemas de Seguranca

| ID | Tipo | Quantidade | Severidade | Area principal | Prioridade | Acao esperada no plano de refatoracao |
| --- | --- | ---: | --- | --- | --- | --- |
| SEG-01 | Execucao dinamica de codigo com `eval()` | 12 | Alta | `demo/`, `test/` | Alta | Substituir por configuracao estruturada, isolar demos ou documentar sandbox. |
| SEG-02 | Manipulacao dinamica de HTML | 32 | Media | `demo/` | Media | Trocar por `textContent`/DOM seguro ou sanitizar HTML dinamico. |
| SEG-03 | Links e recursos HTTP sem TLS | 40 | Media | README, demos, testes | Media | Migrar para HTTPS quando possivel e evitar recursos remotos inseguros. |
| SEG-04 | Dependencia legada jQuery 1.8.3 | 8 referencias | Alta | `demo/` | Alta | Atualizar/remover jQuery ou restringir demos a uso local. |
| SEG-05 | Dependencia legada Jasmine 2.0.0 | 5 referencias | Media | `test/` | Baixa | Planejar atualizacao do runner de testes. |
| SEG-06 | Build com YUI Compressor 2.4.8 | 2 referencias | Media | `compile/` | Media | Documentar origem/hash e avaliar substituto moderno. |
| SEG-07 | Envio opcional de SARIF para servico externo | 4 | Baixa | `.github/` | Baixa | Documentar comportamento e permitir controle explicito por variavel. |

### 3.1 Total por Severidade - Seguranca

| Severidade | Quantidade de categorias | Ocorrencias |
| --- | ---: | ---: |
| Alta | 2 | 20 |
| Media | 4 | 79 |
| Baixa | 1 | 4 |

### 3.2 Interpretacao

Os riscos de seguranca estao concentrados nas demos e no ferramental legado. O nucleo da biblioteca em `src/` nao apresenta, por esta triagem, o mesmo nivel de exposicao direta que as paginas demonstrativas. Ainda assim, como as demos sao parte do escopo educacional do projeto, elas devem ser tratadas no plano de refatoracao.

Prioridades de seguranca:

1. Reduzir ou isolar o uso de `eval()` nas demos.
2. Atualizar ou remover jQuery 1.8.3.
3. Revisar manipulacao dinamica de HTML.
4. Migrar links e recursos HTTP para HTTPS.
5. Modernizar ou documentar melhor ferramentas legadas de teste e build.

## 4. Problemas de Qualidade

| ID | Tipo | Quantidade | Severidade | Area principal | Prioridade | Acao esperada no plano de refatoracao |
| --- | --- | ---: | --- | --- | --- | --- |
| QUA-01 | Uso extensivo de `var` | 1.610 | Media | `src/`, `demo/`, `test/` | Media | Migrar gradualmente para `let`/`const` com testes de regressao. |
| QUA-02 | Comparacoes nao estritas | 293 | Media | `src/`, `demo/`, `test/` | Media | Migrar para `===`/`!==` onde preservar comportamento. |
| QUA-03 | Acoplamento ao namespace global | 110 | Media | `src/`, `build/` | Alta | Mapear API publica e criar organizacao interna menos acoplada. |
| QUA-04 | Aleatoriedade nao controlada | 34 | Media | `src/`, `demo/`, `test/`, `build/` | Alta | Criar mecanismo de aleatoriedade controlavel para testes. |
| QUA-05 | Logs diretos no codigo | 17 | Baixa | `src/`, `demo/`, `test/`, `.github/` | Baixa | Padronizar warnings, erros e logs de demo. |
| QUA-06 | Comentarios TODO/FIXME | 14 | Baixa | `src/`, `demo/`, `test/` | Baixa | Converter TODOs do codigo proprio em backlog rastreavel. |
| QUA-07 | Build e testes legados | 7 referencias criticas | Media | `compile/`, `test/`, exports | Alta | Preservar compatibilidade enquanto se moderniza build/testes. |

### 4.1 Total por Severidade - Qualidade

| Severidade | Quantidade de categorias | Ocorrencias |
| --- | ---: | ---: |
| Alta | 0 | 0 |
| Media | 5 | 2.054 |
| Baixa | 2 | 31 |

### 4.2 Interpretacao

Os problemas de qualidade sao numericamente maiores que os de seguranca porque o projeto possui estilo JavaScript historico, API baseada em namespace global e processo de build por concatenacao. Nem todos devem ser corrigidos de uma vez: o risco de quebrar a API publica e alto, principalmente porque os requisitos exigem compatibilidade com navegador, Node.js, demos e bundle em `build/`.

Prioridades de qualidade:

1. Mapear e proteger a API publica antes de refatorar.
2. Ampliar testes de regressao para `Net`, `Vol`, `Trainer`, camadas e serializacao.
3. Reduzir acoplamento interno em `Net.makeLayers` e `fromJSON`.
4. Controlar aleatoriedade para testes deterministas.
5. Modernizar `var` e comparacoes nao estritas gradualmente.

## 5. Matriz de Priorizacao para Refatoracao

| Prioridade | Problemas relacionados | Justificativa | Fase sugerida |
| --- | --- | --- | --- |
| P1 | SEG-01, SEG-04, QUA-03, QUA-04, QUA-07 | Alto risco de seguranca ou risco estrutural para evolucao segura. | Fases 0, 1, 3, 4, 5 |
| P2 | SEG-02, SEG-03, SEG-06, QUA-01, QUA-02 | Melhorias importantes, mas que exigem testes para evitar regressao. | Fases 1, 2, 5, 6 |
| P3 | SEG-05, SEG-07, QUA-05, QUA-06 | Baixo impacto imediato ou restrito a teste/pipeline/documentacao. | Fases 5, 6 |

## 6. Relacao com o Plano de Refatoracao

| Fase do plano | Problemas contemplados | Resultado esperado |
| --- | --- | --- |
| Fase 0 - Baseline e Inventario | QUA-03, QUA-07 | API publica e demos criticas mapeadas antes de mudancas. |
| Fase 1 - Fortalecimento de Testes | QUA-04, QUA-07, SEG-05 | Rede de seguranca para refatorar codigo legado. |
| Fase 2 - Validacoes e Tratamento de Erros | QUA-05, QUA-06 | Falhas mais previsiveis e menor dependencia de logs soltos. |
| Fase 3 - Separacao Interna de Responsabilidades | QUA-03, QUA-01, QUA-02 | Menor acoplamento e codigo mais legivel. |
| Fase 4 - Refatoracao de Treinadores e Aleatoriedade | QUA-04 | Treinamento e testes mais reproduziveis. |
| Fase 5 - Build, Distribuicao e Compatibilidade | SEG-05, SEG-06, SEG-07, QUA-07 | Build/testes mais auditaveis e compativeis. |
| Fase 6 - Demos e Documentacao Tecnica | SEG-01, SEG-02, SEG-03, SEG-04 | Demos mais seguras e documentadas. |

## 7. Relacao com Casos de Uso

| Caso de uso | Problemas mais relevantes | Impacto |
| --- | --- | --- |
| UC-01 - Definir rede neural | QUA-03, QUA-05, QUA-07 | Afeta manutencao de `makeLayers` e validacoes. |
| UC-02 - Treinar modelo | QUA-04, QUA-05 | Afeta reprodutibilidade e diagnostico de treinamento. |
| UC-03 - Executar inferencia | QUA-03, QUA-07 | Exige preservacao da API publica durante refatoracao. |
| UC-04 - Usar CNN com imagem | SEG-02, QUA-04 | Afeta demos de imagem e manipulacao dinamica de DOM. |
| UC-05 - Serializar e restaurar rede | QUA-03, QUA-07 | Requer testes para nao quebrar JSON de modelos. |
| UC-06 - Executar demo no navegador | SEG-01, SEG-02, SEG-03, SEG-04 | Principal area de risco de seguranca. |
| UC-07 - Compilar biblioteca | SEG-06, QUA-07 | Build legado precisa de controle e documentacao. |
| UC-08 - Rodar testes Jasmine | SEG-05, QUA-07 | Testes precisam ser preservados e depois modernizados. |
| UC-09 - Experimentar Deep Q Learning | SEG-01, QUA-04 | Demo experimental usa execucao dinamica e aleatoriedade. |

## 8. Indicadores para Acompanhar a Refatoracao

| Indicador | Valor inicial | Meta sugerida |
| --- | ---: | --- |
| Ocorrencias de `eval()` | 12 | 0 em demos publicaveis ou uso isolado/documentado |
| Referencias a jQuery 1.8.3 | 8 | 0 ou dependencia atualizada |
| Ocorrencias de HTML dinamico inseguro | 32 | Reduzir ou sanitizar pontos com entrada variavel |
| Links HTTP | 40 | Migrar para HTTPS quando disponivel |
| Ocorrencias de `var` | 1.610 | Reducao gradual por modulo refatorado |
| Comparacoes nao estritas | 293 | Reducao gradual com testes |
| Usos de `Math.random` | 34 | Centralizar aleatoriedade no nucleo e testes |
| Logs diretos | 17 | Manter apenas logs intencionais de demo/pipeline |
| TODO/FIXME | 14 | Converter TODOs proprios em backlog |

## 9. Conclusao

A base de refatoracao deve priorizar seguranca nas demos e qualidade estrutural no nucleo da biblioteca. Embora os problemas de qualidade sejam mais numerosos, os itens de seguranca com `eval()` e jQuery legado merecem tratamento prioritario quando as demos forem publicadas ou usadas fora de ambiente local.

Para reduzir risco de regressao, a ordem recomendada e:

1. inventariar API publica e demos criticas;
2. ampliar testes;
3. corrigir ou isolar riscos de seguranca em demos;
4. reduzir acoplamento interno;
5. controlar aleatoriedade;
6. modernizar build e estilo de codigo gradualmente.

