# Projeto das Lojas Rossmann
![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/rossmann_logo.png)
O Projeto das Lojas Rossmann é um projeto de Ciências de Dados para a predição de vendas da gigante farmaceutica européia Rossmann. 

Esse projeto é uma proposta do curso 'DS em Produção', da Comunidade DS, baseado no desafio 'Rossmann Store Sales' da plataforma Kaggle. O enunciado ofical do problema está disponível em [Kaggle](https://www.kaggle.com/c/rossmann-store-sales).

# 1 Questão de Negócio

A farmaceutica Rossmann é uma rede de farmárcia na Europa que atualmente conta com mais de 4000 lojas atuando na Alemanha, Polônia, Hungria, República Tcheca, Turquia, Albania e Espanha. 

O CFO da empresa fez uma reunião com todos os gerentes de loja e pediu para que cada um deles trouxesse uma previsão diária das próximas seis semanas de vendas. Após essa reunião os gerentes entraram em contato para realizarmos a previsão de vendas de cada uma das lojas.

# 2 Planejamento da Solução

Para solução do desafio dividimos em algumas etapas cíclicas (como mostra a imagem abaixo) de forma que apresentaremos aqui um cíclo completo dessas etapas. Começando na 'Questão de Negócio' até a 'Avaliação do Algoritmo' onde analisaremos a performance do modelo e decidiremos se é necessário realizar mais um ciclo antes de colocar o 'Modelo em Produção'.

![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/Questao%20de%20Negocio%20(1).png)

### 2.1 Produto Final
- Insights de negócio realizados a partir da Análise Exploratória de Dados
- Bot no Telegram que indica a previsão de vendas de qualquer loja

## 2.2 Ferramentas Utilizadas
- Python 3.10
- Jupyter Notebook
- Github
- Heroku 
- Telegram

Vamos entender um pouco melhor como foi cada etapa do projeto.

# 3 Etapas do Projeto

## 3.1 Entendimento do Negócio

Antes de partir para etapas mais técnicas é necessário entender um pouco mais sobre as motivações do problema. Para isso vamos responder quatro perguntas:

- Em que contexto surgiu esse problema?

O problema surgiu na reunião e foi apresentado pelo CFO.

- Por que fazer uma previsão de vendas?

Após conversar com o CFO, ele nos contou que o motivo real da previsão é que ele quer reformar algumas das lojas, mas não sabe quanto de dinheiro deve investir na reforma de cada uma desses lojas. Então ele precisa saber quanto vai vender nas próximas seis semanas, para que adiantar os alocamento de dinheiro que será investido na reforma.

- Quem será o stakeholder?

Podemos ver que o problema é do CFO da empresa e não dos gerentes. Então o CFO, ou alguém muito próximo dele, será o stakeholder.

- Qual é o formato da solução? (Granularidade, tipo de problema, potenciais métodos, formato da entrega)

A granularidade será vendas por dia por loja. O problema é de previsão de vendas. Utilizaremos regressão de séries temporais. Ao final do projeto o CFO  poderá acessar a solução via celular com acesso a interne

## 3.2 Coleta de Dados

Por se tratar de um desafio da plataforma Kaggle, os dados já estão coletados bastando assim acessá-los. Tais dados podem ser acessados pelo [link](https://www.kaggle.com/competitions/rossmann-store-sales/data). Nessa página estão disponíveis quatro arquivos: 

- train.csv: histórico de dados incluindo as vendas
- test.csv: histórico de dados excluindo as vendas
- sample_submission.csv: modelo de submissão do desafio (não utilizaremos)
- store.csv: informações suplementares sobre as lojas

Os atributos apresentados nos conjunto de dados são:

| Atributo                          | Descrição |  
| --------------------------------  | --------- |
| ID                                | Representa uma loja em  uma data específica que foram relaizadas as vendas |   
| Store                             | Número único de cada loja |
| Sales                             | Total de vendas realizada em dia |
| Date                              | Data |
| DayOfWeek                         | Dia da semana |
| Customers                         | Número de clientes de um dia |
| Open                              | Indica se a loja estava ou não aberta naquela data (0: fechada ou 1: aberta) |
| StateHoliday                      | Indica um feriado de estado (a: public holiday, b: Easter holiday, c: Christmas, 0: None) |
| SchoolHoliday                     | Indica se a loja foi afetada pelo fechamento das escolas públicas naquela data |
| StoreType                         | Diferencia as lojas em quatro tipos: a, b, c, d |
| Assortment                        | Descreve o nível de estoque das lojas (a: basic, b: extra, c: extended) |
| CompetitionDistance               | Distância em metros do competidor mais pŕoximo |
| CompetitionOpenSince[Month/Year]  | A data (mês/ano) aproximada de quando o competidor mais próximo abriu |
| Promo                             | Indica se a lojas está com promoção no dia |
| Promo2                            | É uma promoção contínua e consecutiva para algumas lojas (0: não está participando, 1: está participando) |
| Promo2Since[Year/Week]            | Descreve a semana e ano em que a loja começou a participar da Promo2 |
| PromoInterval                     | Descreve os intervalos consecutivos em que a Promo2 é iniciada, nomeando os meses em que a promoção é iniciada novamente |

Agora que os dados estão coletados partiremos para a limpeza dos dados.

## 3.3 Limpeza dos Dados

Nessa etapa começamos a parte mais técnica do projeto, utilizando Python fizemos algumas mudanças nos conjuntos de dados de modo a torna-lo mais funcional para etapas futuras, como a Análise Exploratória dos Dados e a implementação dos Modelos de Machine Learning. Algumas dessas mudanças foram:

- lojas sem CompetitionDistance foram consideradas sem competidores próximos, então completamos com 200000 que é um valor muito superior aos outros;
- lojas sem CompetitionOpenSince[Month/Year] também foram consideradas sem competidores próximos, então completamos com a própria data da colunas 'date';
- o mesmo foi realizado para lojas sem Promo2Since[Year/Week];
- lojas sem venda ou que estavam fechadas em determinado dia foram desconsideradas no treinamento do modelo.


## 3.4 Análise Exploratória dos Dados

Aqui começamos fazendo o Mapa Mental de Hipóteses abaixo. Fizemos as análises univariadas, bivariadas e multivariadas. Para as bivariadas, redigimos algumas hipóteses de negócios e verificamos a veracidade delas. 

![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/mindmap_hypoteses.png)

### 3.4.1 Análise Univariada



#### 3.4.1.1 Variável Resposta

![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/variavel_resposta.png)

#### 3.4.1.2 Variáveis Numéricas

![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/univariada_variaveis_numericas.png)

#### 3.4.1.3 Variáveis Categóricas

![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/univariada_variaveis_categoricas.png)

### 3.4.2 Análise Bivariada
Analisamos várias hipóteses, aqui apresentaremos apenas os cinco Insights mais interessantes

- Lojas deveriam vender mais no segundo semestre do ano.

![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/bivariada_1.png)

Verdadeiro. Lojas vendem mais no segundo semestre.

-  Lojas abertas durante o feriado de Natal deveriam vender mais.

![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/bivariada_2.png)

Verdadeiro. Lojas vendem, na média, mais durante o feriado de Natal e de Páscoa.

- Lojas com maior sortimentos deveriam vender mais.

![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/bivariada_3.png)

Verdadeiro. Lojas com maior sortimento vendem mais, na média.

- Lojas com mais promoções consecutivas deveriam vender mais.

![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/bivariada_4.png)

Falso. Lojas com mais promoções consecutivas vendem menos.

- Lojas deveriam vender menos aos finais de semana.

![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/bivariada_5.png)

Verdadeiro. Lojas vendem menos nos finais de semana.

- Lojas deveriam vender menos durante os feriados escolares.

![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/bivariada_6.png)

Falso. Na média, lojas vendem mais durante os feriados escolares, com exceção dos meses de setembro e dezembro.



## 3.5 Modelagem dos Dados

Começamos essa etapa com a preparação dos dados para a implementação dos modelos de Machine Learning. Para os dados númericos e não ciclícos utilizamos algumas estratégias de *rescaling* como *RobustScaler* e *MinMaxScaler*. Já para os dados categóricos fizemos o *encoding* dessas variáveis, entre as estratégias utilizadas estão: *One Hot Encoding*, *Label Encoding* e *Ordinal Encoding*. Para a variável resposta ('sales') fizemos uma transformação logarítimica e para as variáveis de natureza cíclica realizamos transformações trigonométricas.

Após as transformações das variáveis, é necessário selecionar os melhores atributos para o treino dos modelos de ML. Para isso usamos o algoritmo Boruta, que é um métodos baseado Random Forest e funcionamento muito bem com modelos de árvore como Random Forest e XGBoost.

## 3.6 Algoritmos de Machine Learning

Agora com as variáveis ajustadas e selecionadas, estamos prontos para aplicar os algoritmos de Machine Learning. Nesse projeto testamos cinco modelos: 

- Modelo de Média
- Modelo de Regressão Linear
- Modelo de Regressão Linear Regularizado (Lasso)
- Modelo de Regressão Random Forest
- Modelo de Regressão XGBoost

Para cada modelo calculamos três erros:

- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentual Error)
- RMSE (Root Mean Square Error)

Os erros encontrados para cada modelos foram:

| Modelo                        | MAE        | MAPE    | RMSE       |
|-------------------------------|------------|---------|------------|
| Média                         | 1354.80035 | 0.20640 | 1835.13554 |
| Regressão Linear              | 1870.16131 | 0.29509 | 2667.88331 |
| Regressão Linear Regularizado | 1891.19496 | 0.28987 | 2741.00328 |
| Regressão Random Forest       | 723.53226  | 0.10699 | 1077.12693 |
| Regressão XGBoost            | 871.06206	 | 0.12813 | 1271.58178 |

Após o cross-validation os erros foram:

| Modelo                        | MAE                   | MAPE              | RMSE                   |
|-------------------------------|-----------------------|-------------------|------------------------|
| Regressão Linear              | 1925.3411 +/- 39.4565 |	0.2917 +/- 0.0058 |	2758.5347 +/- 71.9813  |
| Regressão Linear Regularizado | 1958.2725 +/- 49.3824 |	0.2865 +/- 0.0023	| 2848.217 +/- 78.8584   |
| Regressão Random Forest       | 790.9915 +/- 81.7962  |	0.1126 +/- 0.0114 |	1201.1863 +/- 150.5484 |
| Regressão XGBoost             | 961.176 +/- 66.5849   |	0.134 +/- 0.007	  | 1403.1786 +/- 96.2286  |

O melhor resultado deu-se pela Regressão Random Forest, porém por questão de estudo aplicaremos a Regressão XGBoost que não teve um resultado tão abaixo mas que possui um custo de armazenamento muito menor que o da Random Forest.

Após todos esses cálculos também realizamos a técnica 'Hyperparameter Fine Tunning' para a otimização dos parâmetros do modelo de ML e o resultado dos erros foram:

| Modelo                        | MAE                   | MAPE              | RMSE                   |
|-------------------------------|-----------------------|-------------------|------------------------|
| Regressão XGBoost             | 758.6882 +/- 56.546   |	0.1098 +/- 0.0038 | 1085.653 +/- 93.4135   |

## 3.7 Avaliação do Algoritmo

### 3.7.1 Performance de Negócio
Finalmente, com o modelo treinado e com os erros calculados, podemos calcular a estimativa de vendas da empresa. Realizando a soma de todas as previsões de vendas cehgamos no valore de $ 289.741.984,00. Porém, é necessário considerar o erro de  aproximadamente 10% (MAPE), chegando nos seguintes resultados:

| Cenário        | Previsão        |
|--------        | ---------       |
| Pior Cenário   |$ 259.377.024,00 |
| Média          |$ 289.741.984,00 |
| Melhor Cenário |$ 320.106.944,00 |

### 3.7.2 Performance do Modelo

Abaixo seguem quatro gráfico que resumem a avaliação do modelo de Machine Learning.

![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/avaliacao_erro.png)

No primeiro gráfico, vemos o quão próximas estão as curvas de previsão e de vendas. O segundo é um gráfico da data x taxa de erro, onde a linha tracejada indica que a previsão foi igual a venda, nota-se que a taxa foi no máximo 15% superior e no mínimo 15% inferior. O terceiro gráfico é um histograma dos erros (diferença entre a previsão e o valor real), notamos que é muito próxma da curva normal, o que indica um ótimo resultado. E no último vemos que a distribuição das previsão com seus erros formam um "tubo", com poucos *outliers*, o que também caracteriza um bom resultado.

## 3.8 Deploy do Modelo em Produção

Para que o CFO tivesse todas as informações sobre as lojas de qualquer lugar com acesso a internet, configuramos um Bot no Telegram que nos da a previsão de vendas de qualquer loja em dois formatos: a previsão diária durante as próximas seis semanas e a previsão total dessas seis semanas. O deploy foi realizado no Heroku.

https://user-images.githubusercontent.com/102927918/180664797-ee82a775-94f0-48fa-ba0b-b7cfc82da5b9.mp4

# 4 Conclusão

O projeto foi concluído com sucesso, conseguimos prever a venda de cada um das lojas com um erro média de 10%. Esse calculos nos permitiu prever que a venda total de todas as lojas Rossmann seria em torno de $ 290 milhões. Para esses cálculos modelamos o problema com uma Regressão XGBoost que obteve uma performance muito boa.

A Análise Exploratória dos Dados se mostrou muito rica para o entendimento o projeto e com certeza é uma etapa que não deve ser descartada, mesmo que interfira diretamente no resultado o modelo.

Com os resultados obtidos foi possível colocar o projeto em produção já na primeira volta do ciclo de etapas. O deploy foi realizado no Heroku e o Bot do Telegram funciona em qualquer lugar com acesso a internet.


# 5 Próximos Passos

Após o primeiro ciclo das etapas novos ciclos devem ser realizados, afim que novos Insights possam ser gerados na etapa de Análise Exploratória dos Dados ou que alguns atributos possam ser acrescentados ou retirados do treinamento do modelo de ML. 


# 6 Referências
[1] LOPES, Meigarom. Curso DS em Produção - Comunidade DS.
[2] Wikpédia. Cramér's V, https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V, último acesso: 24/07/2022
[3] ROY, Baijayanta, All about Categorical Variable Encoding, https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02, último acesso: 24/07/2022


