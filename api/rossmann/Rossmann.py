import pickle
import inflection
import pandas as pd
import numpy as np
import datetime



class Rossmann(object):
    def __init__ (self):
        self.competition_distance_scaler   = pickle.load(open('/home/jonas/Documentos/repos/ds_em_producao/parameter/competition_distance_scaler.pkl', 'rb'))
        self.year_scaler                   = pickle.load(open('/home/jonas/Documentos/repos/ds_em_producao/parameter/year_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open('/home/jonas/Documentos/repos/ds_em_producao/parameter/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler        = pickle.load(open('/home/jonas/Documentos/repos/ds_em_producao/parameter/promo_time_week_scaler.pkl', 'rb'))
        self.store_type_encoding           = pickle.load(open('/home/jonas/Documentos/repos/ds_em_producao/parameter/store_type_encoding.pkl','rb'))
        
        
    def data_cleaning(self, df1):
        # renomeando as colunas
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo','StateHoliday', 'SchoolHoliday', 
            'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth','CompetitionOpenSinceYear', 
            'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

        cols_new = []

        for c in cols_old:
            cols_new.append(inflection.underscore(c))
    
        df1.columns = cols_new
        
        # date to datetime
        df1['date'] = pd.to_datetime(df1['date'])

        # preenchendo os NA's
        # competition_distance
        # as distância que não existem vamos colocar 200000, que é um valor muito maior que a maior distância
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 200000 if np.isnan(x) else x)

        # competition_open_since_month    
        # vamos completar os Na's com o mês da coluna 'date'
        df1['competition_open_since_month'] = df1[['date','competition_open_since_month']].apply(lambda x: x['date'].month
                                                                                                if np.isnan(x['competition_open_since_month'])
                                                                                                else x['competition_open_since_month'], axis=1)

        # competition_open_since_year   
        # vamos completar os Na's com o ano da coluna 'date'
        df1['competition_open_since_year'] = df1[['date','competition_open_since_year']].apply(lambda x: x['date'].year
                                                                                                if np.isnan(x['competition_open_since_year'])
                                                                                                else x['competition_open_since_year'], axis=1)


        # promo2_since_week      

        df1['promo2_since_week'] = df1[['date','promo2_since_week']].apply(lambda x: x['date'].week
                                                                           if np.isnan(x['promo2_since_week'])
                                                                           else x['promo2_since_week'], axis=1)

        # promo2_since_year   

        df1['promo2_since_year'] = df1[['date','promo2_since_year']].apply(lambda x: x['date'].year
                                                                           if np.isnan(x['promo2_since_year'])
                                                                           else x['promo2_since_year'], axis=1)

        # promo_interval  
        df1['promo_interval'].fillna(0, inplace=True)

        month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
        df1['month_map'] = df1['date'].dt.month.map(month_map)


        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 
                                                                           else 1 if x['month_map'] in x['promo_interval'].split(',')
                                                                           else 0, axis=1)
        # mudando o tipo das variáveis
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)

        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

        return df1
        
    def feature_engineering(self, df2):
        # feature engineering
        # year
        df2['year'] = df2['date'].dt.year

        # month
        df2['month'] = df2['date'].dt.month

        # day
        df2['day'] = df2['date'].dt.day

        # week of year
        df2['week_of_year'] = df2['date'].dt.isocalendar().week
        df2['week_of_year'] = df2['week_of_year'].astype('int64') 

        # formato da data: Year-Week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        # competition since
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'],
                                                                         month=x['competition_open_since_month'],
                                                                         day=1), axis=1)
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).dt.days.astype(int)

        # promo since
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str) + '-1'
        df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%W-%w') - datetime.timedelta(days=7))
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).dt.days

        # assortment
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else
                                                              'extra' if x == 'b' else
                                                              'extended')

        # state holiday
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 
                                                                    'easter' if x == 'b' else
                                                                    'christmas' if  x == 'c' else
                                                                    'regular_day')

        # filtragem de variáveis
        # Filtrar apenas as lojas abertas e que obtiveram vendas.
        df2 = df2[(df2['open'] != 0)]
        
        # Algumas colunas que não fazem mais sentido, ou que substituimos por outras.
        cols_drop = ['open', 'promo_interval', 'month_map']
        df2 = df2.drop(cols_drop, axis=1)
        
        return df2
    
    def data_preparation(self, df5):
    
        # competition_distance (analisando o boxplot vemos que possui muitos valores atípicos (outliers), então vamos utilizar o RobustScaler)
        df5['competition_distance'] = self.competition_distance_scaler.transform(df5[['competition_distance']])

        # year (analisando o boxplot vemos que não possui muitos valores atípicos, então vamos utilizar o MinMaxScaler)
        df5['year'] = self.year_scaler.transform(df5[['year']])

        # competiton_time_month (analisando o boxplot vemos que possui muitos valores atípicos (outliers), então vamos utilizar o RobustScaler)
        df5['competition_time_month'] = self.competition_time_month_scaler.transform(df5[['competition_distance']])

        # promo_time_week (analisando o boxplot vemos que não possui muitos valores atípicos, então vamos utilizar o MinMaxScaler)
        df5['promo_time_week'] = self.promo_time_week_scaler.transform(df5[['promo_time_week']])


        # state holiday (One Hot Encoding)
        df5 = pd.get_dummies(data=df5, prefix=['state_holiday'], columns=['state_holiday'])

        # store type (Label Encoding)
        df5['store_type'] = self.store_type_encoding.transform(df5['store_type'])

        # assortment (Ordinal Encoding, basic = 1, extra = 2, extended = 3)
        assortment_dic = {'basic':1, 'extra':2, 'extended':3}
        df5['assortment'] = df5['assortment'].map(assortment_dic)


        # variáveis cíclicas
        # day_of_week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x*2*np.pi/7))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x*2*np.pi/7))
        # month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x*2*np.pi/12))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x*2*np.pi/12))
        # day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x*2*np.pi/30))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x*2*np.pi/30))
        # week_of_year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x*2*np.pi/52))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x*2*np.pi/52))
        
        # colunas selecionadas para o modelo de ML
        cols_selected = ['store',
                         'promo',
                         'assortment',
                         'competition_distance',
                         'competition_open_since_month',
                         'competition_open_since_year',
                         'promo2',
                         'promo2_since_week',
                         'promo2_since_year',
                         'competition_time_month',
                         'promo_time_week',
                         'store_type',
                         'day_of_week_sin',
                         'day_of_week_cos',
                         'month_cos',
                         'day_cos',
                         'week_of_year_cos']
        
        return df5[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data)
                
        # join pred into the original data
        original_data['predictions'] = np.expm1(pred)
                
        return original_data.to_json(orient='records', date_format='iso')
