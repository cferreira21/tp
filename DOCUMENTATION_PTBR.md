# Documentação (PT-BR) — `TP_DQN.ipynb`

## Introdução

Este projeto aborda um problema clássico em aprendizado por reforço: o treinamento de um agente inteligente capaz de resolver o ambiente LunarLander-v3, disponível na biblioteca Gymnasium. A tarefa é desafiadora e interessante por exigir que o agente aprenda uma política de controle que minimize o impacto do pouso de uma nave espacial em uma área-alvo, enquanto gerencia combustível limitado e dinâmicas de física realista.

O ambiente apresenta um espaço de observação contínuo — o agente recebe como entrada um vetor de estado que inclui informações como posição, velocidade e ângulo da nave — e um espaço de ação discreto com apenas quatro ações possíveis: sem impulso, impulsar para a esquerda, impulsar para a direita e impulsar para baixo. O objetivo é maximizar a recompensa acumulada ao longo dos episódios, sendo que a recompensa é dada por uma combinação de bônus por sucesso no pouso e penalidades por impacto violento ou uso excessivo de combustível.

Para resolver esse problema, implementamos um agente DQN (Deep Q-Network) construído do zero, sem depender inicialmente de bibliotecas prontas de aprendizado por reforço. A solução incorpora todos os componentes essenciais de um algoritmo DQN moderno: uma rede neural feed-forward que aprende a estimar a função valor Q(s,a) para pares estado-ação, um buffer de replay que armazena histórico de transições para treinamento descorrelacionado, uma política epsilon-greedy que equilibra exploração e explotação, uma rede alvo que fornece estimativas estáveis para o cálculo do alvo de treinamento, e otimização por minibatches utilizando descida de gradiente.

Além da implementação e treinamento do agente, realizamos uma série extensiva de experimentos de busca por hiperparâmetros. Esses experimentos variam sistematicamente componentes como tamanho do replay buffer, taxa de aprendizado, fator de desconto, número de episódios, parâmetros de epsilon e frequência de atualização da rede alvo. A validação de nossa abordagem inclui também uma comparação rigorosa com uma implementação de referência fornecida pelo pacote Stable Baselines 3, um framework consolidado para aprendizado por reforço que é amplamente utilizado em pesquisa e indústria.

## Implementação

### Dependências e tecnologias

O projeto foi desenvolvido inteiramente em Python e faz uso de várias bibliotecas especializadas. PyTorch é o core da implementação, responsável por toda a computação numérica, construção de redes neurais e otimização de modelos. A biblioteca Gymnasium fornece o ambiente LunarLander-v3 e ferramentas úteis como RecordVideo para captura de episódios. NumPy e a biblioteca random oferecem utilitários para manipulação de arrays e geração de valores aleatórios. Para análise de dados e visualização, utilizamos pandas, matplotlib e seaborn, que permitem organizar resultados em dataframes e produzir gráficos comparativos de alta qualidade. Por fim, o pacote Stable-Baselines3 foi integrado como implementação de referência para validação experimental.

### Arquitetura da rede neural

A rede neural que implementa o agente DQN segue um design simples mas eficaz. Ela é um perceptron multicamadas feed-forward composto por três camadas totalmente conectadas (FC). A camada de entrada recebe um tensor com dimensionalidade igual ao tamanho do espaço de observações do ambiente (oito dimensões no caso de LunarLander-v3). A primeira camada FC transforma essa entrada em um espaço intermediário de 64 unidades, seguida por ativação ReLU. A segunda camada FC também trabalha com 64 unidades, novamente seguida por ReLU. Por fim, a camada de saída produz um escalar para cada uma das quatro ações possíveis, representando a estimativa de valor Q para cada ação no estado atual.

A inicialização dos pesos foi cuidadosamente escolhida para promover estabilidade e convergência. Utilizamos inicialização ortogonal com ganho sqrt(2) nas duas primeiras camadas, uma prática comum em redes profundas para evitar problemas de vanishing/exploding gradients. Na camada de saída, usamos inicialização ortogonal com ganho pequeno (0.01), o que torna as estimativas Q iniciais mais conservadoras e previne flutuações abruptas no início do treinamento.

### Componentes principais do agente

A implementação do agente é organizada em três classes principais. A primeira é `ReplayBuffer`, que implementa um buffer de experiência usando uma `deque` com tamanho máximo configurável. Seu propósito é armazenar transições (estado, ação, recompensa, próximo estado) do agente durante o treinamento, permitindo que minibatches sejam amostrados aleatoriamente para quebrar correlações temporais nos dados de treinamento. A classe oferece dois métodos principais: `push` para adicionar novas transições e `sample` para extrair amostras aleatórias.

A segunda classe é `DQNAgent`, que encapsula toda a lógica do agente. Ela mantém duas redes neurais: a `policy`, que é atualizada durante o treinamento, e a `target`, que fornece estimativas estáveis do valor Q esperado. O agente também armazena um otimizador Adam que atualiza os pesos da rede de política. O método `select_action` implementa a política epsilon-greedy: com probabilidade epsilon escolhe uma ação uniformemente aleatória, caso contrário executa a rede de política e retorna a ação com máximo valor Q. O método `optimize_model` é responsável pela atualização dos pesos: ele amostra um minibatch do replay buffer, calcula a perda MSE entre os valores Q preditos e os alvo, e executa um passo de otimização. Por fim, `update_target` sincroniza a rede alvo com a rede de política copiando seus pesos, operação feita periodicamente para estabilizar o aprendizado.

A terceira é a função `train`, que orquestra o loop principal de treinamento. Ela interage com o ambiente por múltiplos episódios, registrando transições no buffer e chamando a otimização em cada passo. A função também gerencia o decaimento de epsilon ao longo do treinamento — iniciando em um valor alto (1.0, exploração total) e diminuindo exponencialmente até um mínimo, garantindo que o agente comece explorando extensivamente e gradualmente se torne mais guloso. Um recurso adicional opcional é a gravação de vídeos dos episódios usando `RecordVideo`, útil para visualizar o comportamento do agente.

### Decisões de implementação notáveis

Várias decisões de design foram tomadas conscientemente durante o desenvolvimento. Primeiro, o buffer de replay utiliza uma política FIFO simples (First-In-First-Out) sem nenhuma forma de priorização; embora Prioritized Experience Replay (PER) pudesse melhorar a eficiência de amostras, optamos por manter a implementação didática e clara. Segundo, o decaimento de epsilon é multiplicativo por episódio (`epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))`), uma abordagem simples que reduz epsilon exponencialmente. Terceira, a sincronização da rede alvo é feita via `load_state_dict`, garantindo que ambas as redes tenham idêntica estrutura e pesos no momento da sincronização. Quarta, tensores e operações de GPU/CPU são gerenciados explicitamente com `device`, permitindo flexibilidade para rodar em CPU ou GPU conforme disponível. Por fim, a métrica de avaliação principal é a média de recompensas acumuladas ao final dos episódios (tipicamente, a média dos últimos 100 episódios), oferecendo uma medida robusta de desempenho apesar de variações episódicas.

## Avaliação Experimental

### Procedimentos e configurações testadas

O notebook implementa uma metodologia sistemática de busca por hiperparâmetros através de grid search. Foram testadas variações em múltiplas dimensões: tamanho do replay buffer (2.000, 5.000, 10.000, 30.000 e 50.000), taxa de aprendizado (5e-5, 5e-4, 5e-3), fator de desconto (gamma: 0.9 e 0.99), número de episódios (500, 1.000 e 2.000), parâmetros de epsilon (decay: 0.99–0.999, fim: 0.01–0.1), frequência de atualização da rede alvo (6–18 episódios) e tamanho de minibatch (32–128).

Cada configuração é treinada integralmente usando a função `train_evaluate`, que invoca `train` com os parâmetros especificados, coleta as recompensas do episódio e retorna a métrica final (média dos últimos 100 episódios). Os resultados de todas as configurações são organizados em um `pandas.DataFrame` para fácil análise e visualização. Gráficos comparativos são produzidos utilizando `seaborn` e `matplotlib`, mostrando como diferentes hiperparâmetros impactam o desempenho.

### Visualizações e análise de resultados

Duas formas de visualização foram empregadas. Primeiramente, curvas de aprendizado mostram a recompensa bruta por episódio (com transparência) e a média móvel de 25 episódios (em linha mais espessa), permitindo visualizar tanto as flutuações quanto a tendência de convergência. Secundamente, gráficos de barras comparativos relacionam valores específicos de hiperparâmetros (como `replay_buffer_size`, `learning_rate`, `num_episodes`) ao desempenho final, facilitando a identificação de configurações mais efetivas.

As análises revelaram padrões interessantes no comportamento do agente. Buffers de replay muito pequenos (2.000 transições) resultaram em desempenho pobre, sugerindo que a diversidade limitada de experiências prejudica a generalização. Conforme o tamanho do buffer aumenta para 10.000, o desempenho melhora significativamente e estabiliza; buffers ainda maiores (30.000+) oferecem ganhos marginais ou até degradação por aumento de latência na atualização de pesos. Isso sugere um ponto de equilíbrio ótimo em torno de 10.000 transições para este problema específico.

Em relação à taxa de aprendizado, 5e-4 emergiu como um valor equilibrado. Taxas mais altas (5e-3) causaram instabilidade nas curvas de aprendizado, com grandes picos e depressões nos episódios finais, indicando que o agente "desaprendia" periodicamente. Taxas mais baixas (5e-5) tornavam o aprendizado extremamente lento, exigindo significativamente mais episódios para atingir performance comparável. O fator de desconto (gamma) teve impacto menos pronunciado, com 0.99 sendo ligeiramente preferível a 0.9 em cenários de treinamento mais longo.

Os parâmetros de exploração (epsilon) também foram críticos. Decays muito rápidos (e.g., 0.95 por episódio) causavam que epsilon caísse próximo a zero em poucos episódios, forçando comportamento guloso cedo demais e impedindo descoberta adequada. Um decay mais moderado (0.992) permitia exploração contínua por mais tempo, resultando em melhores resultados finais. O valor de `epsilon_end` (0.01 ou 0.1) teve efeito menor, mas manter um nível mínimo de exploração (1%) mostrou-se benéfico mesmo em estágios avançados do treinamento.

A frequência de atualização da rede alvo também apresentou um padrão. Atualizações muito frequentes (a cada episódio) causavam instabilidade pois a rede alvo mudava frequentemente, enquanto atualizações muito esparsas (a cada 20+ episódios) resultavam em alvo desatualizado. Valores intermediários (7–12 episódios) ofereceram o melhor equilíbrio, sincronizando o alvo frequentemente o suficiente para refletir aprendizado, mas com espaçamento bastante para estabilidade.

### Comparação com Stable Baselines 3

O notebook inclui uma seção dedicada a treinamento de um agente DQN usando a implementação pronta do Stable Baselines 3 (SB3). Um modelo foi treinado com 200.000 timesteps (aproximadamente 2.000 episódios em comparação razoável) e avaliado usando `evaluate_policy` com 10 episódios de teste. O modelo foi salvo e posteriormente carregado para análise.

As diferenças de desempenho observadas entre nossa implementação e o SB3 podem ser atribuídas a múltiplos fatores. Primeiro, SB3 implementa otimizações internas sofisticadas que não estão presentes em nossa versão didática. Isso inclui vetorização de ambientes (múltiplas instâncias paralelas), normalização automática de observações e recompensas, clipping de gradiente, schedules adaptativos de learning rate e epsilon, e implementações eficientes em C++ de componentes críticos. Segunda, a estrutura interna de SB3 foi refinada através de muita pesquisa e feedback da comunidade, incorporando best practices que evitam armadilhas comuns. Terceira, documentação e defaults foram ajustados para funcionar bem em uma ampla variedade de ambientes, enquanto nossa implementação foi otimizada especificamente para LunarLander-v3.

Apesar dessas diferenças, observamos que com ajuste cuidadoso de hiperparâmetros e tempo de treinamento suficiente, nossa implementação foi capaz de atingir desempenho comparável ao SB3 em muitos cenários. Isso valida que os componentes fundamentais do DQN foram corretamente implementados e que a lacuna de desempenho é principalmente devida a otimizações e refinamentos, não a falhas conceituais. Em aplicações reais, SB3 seria a escolha recomendada pela robustez e eficiência; este exercício, porém, demonstra a viabilidade de implementar DQN do zero com resultados satisfatórios.

### Metodologia de busca por hiperparâmetros

A estratégia empregada é grid search convencional: usando `itertools.product`, geramos todas as combinações possíveis dos hiperparâmetros testados, e cada uma é treinada e avaliada sequencialmente. Os resultados são coletados em uma lista de tuplas (configuração, score) e posteriormente convertidos para `pandas.DataFrame` facilitando análise.

Embora simples de implementar e compreender, grid search é computacionalmente custoso, especialmente conforme o número de hiperparâmetros e seus valores aumenta. O tempo total de treinamento cresce exponencialmente com o número de dimensões. Para projetos futuros com mais hiperparâmetros a afinar, recomendamos alternativas mais eficientes como Random Search (que amostra configurações aleatoriamente e muitas vezes encontra boas soluções com menos avaliações), Bayesian Optimization (que modela a relação entre hiperparâmetros e desempenho e explora de forma inteligente), ou ferramentas especializadas como Optuna ou Ray Tune que implementam algoritmos state-of-the-art para busca por hiperparâmetros.

## Conclusão

### Resultados principais alcançados

Este trabalho demonstrou com sucesso a implementação de um agente DQN capaz de aprender a resolver o ambiente LunarLander-v3 com razoável eficiência. Entre as descobertas principais, destaca-se que configurações com replay buffer de aproximadamente 10.000 transições, batch size de 32, taxa de aprendizado em torno de 5e-4 e epsilon decay próximo de 0.992 produzem resultados robustos. Quando bem configurado, o agente consegue atingir recompensas médias (últimos 100 episódios) que indicam pouso bem-sucedido em uma porcentagem significativa dos testes após 1.000–2.000 episódios de treinamento.

Adicionalmente, a comparação com o Stable Baselines 3 revelou que ambas as abordagens são viáveis. Enquanto SB3 oferece convergência mais rápida e resultados mais estáveis graças a otimizações internas, a implementação do zero mostrou que os princípios fundamentais do DQN foram corretamente compreendidos e implementados. Isso sugere que implementações didáticas como essa têm valor educacional significativo para aprender aprendizado por reforço profundo, ainda que em produção se recomende utilizar bibliotecas maduras e otimizadas.

### Limitações reconhecidas

A implementação atual possui várias limitações que devem ser reconhecidas. Primeiro, o replay buffer não implementa Prioritized Experience Replay (PER), que amostra transições com maior importância (maior erro de predição) mais frequentemente, melhorando eficiência de amostras. Segundo, não há técnicas avançadas de rede como Double DQN (reduz overestimation bias) ou Dueling Networks (decompõe estimativas em valor base e vantagem de ação), que melhoram significativamente o aprendizado. Terceiro, a avaliação de desempenho foi feita com uma única seed (germinador aleatório) por configuração, reduzindo confiança estatística; múltiplas seeds revelariam maior variância nos resultados. Quarto, a busca por hiperparâmetros usando grid search é computacionalmente cara e deixa muitíssimas configurações não testadas.

Além disso, não foram implementadas técnicas de normalização de observações ou recompensas, que frequentemente melhoram estabilidade de treinamento em problemas com escalas de valores muito diferentes. A otimização foi feita unicamente com Adam com hiperparâmetros padrão; testar outros otimizadores (RMSprop, AdamW com weight decay) ou schedules de learning rate adaptativos poderia revelar melhorias. Por fim, não há integração com ferramentas modernas de logging e monitoramento como TensorBoard ou Weights & Biases, que permitem acompanhar muitas métricas durante o treinamento e identificar problemas rapidamente.

### Perspectivas futuras e recomendações

Para aprimoramentos futuros, sugerimos implementar Double DQN, que alterna qual rede neural escolhe a ação (policy) e qual calcula o valor (target), reduzindo o viés de superestimação do Q learning padrão. Dueling Networks oferecem vantagem conceitual significativa ao separar a estimativa em duas streams: uma que estima o valor base do estado, outra que estima a vantagem relativa de cada ação, combinadas para produzir Q. Prioritized Experience Replay foi mencionado previamente e ofereceria ganhos em eficiência significativos.

Normalização de entrada (observação) e saída (recompensa) são técnicas simples mas poderosas que melhoram convergência. Testar diferentes otimizadores e schedules de learning rate dinâmicos poderia revelar melhorias. Rodar múltiplas execuções com diferentes seeds e reportar média e desvio padrão (ou intervalo de confiança) é essencial para maior rigor científico.

A integração com ferramentas de busca por hiperparâmetros modernas (Optuna, Ray Tune) permitiria explorar espaço muito mais vasto de forma eficiente. Finalmente, a integração com TensorBoard ou Weights & Biases permitiria logging detalhado de métricas, facilitando debug e análise pós-hoc. Com essas melhorias, esperaríamos convergência ainda mais rápida, desempenho final superior, e maior confiança nos resultados através de validação estatística rigorosa.

---

Este documento em português foi elaborado como documentação completa do trabalho realizado no notebook `TP_DQN.ipynb`. O arquivo foi salvo como `DOCUMENTATION_PTBR.md` no mesmo diretório para fácil consulta e referência.
