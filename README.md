# Projeto das Lojas Rossmann
![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/rossmann_logo.png)
O Projeto das Lojas Rossmann é um projeto de Ciências de Dados para a predição de vendas da gigante farmaceutica européia Rossmann. 

Esse projeto é uma proposta do curso 'DS em Produção', da Comunidade DS, baseado no desafio 'Rossmann Store Sales' da plataforma Kaggle. O enunciado ofical do problema está disponível em [Kaggle](https://www.kaggle.com/c/rossmann-store-sales).

Para solução do desafio dividimos em algumas etapas cíclicas (como mostra a imagem abaixo) de forma que apresentaremos aqui um cíclo completo dessas etapas. Começando na 'Questão de Negócio' até a 'Avaliação do Algoritmo' onde analisaremos a performance do modelo e decidiremos se é necessário realizar mais um ciclo antes de colocar o 'Modelo em Produção'.

![alt text](https://github.com/jonasbarletta/ds_em_producao/blob/main/img/Questao%20de%20Negocio%20(1).png)

# 1 Questão de Negócio

A farmaceutica Rossmann é uma rede de farmárcia na Europa que atualmente conta com mais de 4000 lojas atuando na Alemanha, Polônia, Hungria, República Tcheca, Turquia, Albania e Espanha. 

O CFO da empresa fez uma reunião com todos os gerentes de loja e pediu para que cada um deles trouxesse uma previsão diária das próximas seis semanas de vendas. Após essa reunião os gerentes entraram em contato para realizarmos a previsão de vendas de cada uma das lojas.

# 2 Entendimento do Negócio

Antes de partir para etapas mais técnicas é necessário entender um pouco mais sobre as motivações do problema. Para isso vamos responder quatro perguntas:

- Em que contexto surgiu esse problema?

O problema surgiu na reunião e foi apresentado pelo CFO.

- Por que fazer uma previsão de vendas?

Após conversar com o CFO, ele nos contou que o motivo real da previsão é que ele quer reformar algumas das lojas, mas não sabe quanto de dinheiro deve investir na reforma de cada uma desses lojas. Então ele precisa saber quanto vai vender nas próximas seis semanas, para que adiantar os alocamento de dinheiro que será investido na reforma.

- Quem será o stakeholder?

Podemos ver que o problema é do CFO da empresa e não dos gerentes. Então o CFO, ou alguém muito próximo dele, será o stakeholder.

- Qual é o formato da solução? (Granularidade, tipo de problema, potenciais métodos, formato da entrega)

A granularidade será vendas por dia por loja. O problema é de previsão de vendas. Utilizaremos regressão de séries temporais. Ao final do projeto o CFO poderá acessar a solução via celular com acesso a internet.

# 3 Coleta de Dados

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
| Promo2Since[Year/Week]            | Descreve a semana do ano em que a loja começou a participar da Promo2 |
| PromoInterval                     | Descreve os intervalos consecutivos em que a Promo2 é iniciada, nomeando os meses em que a promoção é iniciada novamente |

Agora que os dados estão coletados partiremos para a limpeza dos dados.

# 4 Limpeza dos Dados

