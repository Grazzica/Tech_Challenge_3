import pandas as pd
import numpy as np 
from pathlib import Path

COLUMNS_TO_DROP = ['YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'SCHEDULED_ARRIVAL', 'SCHEDULED_TIME', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
                   'WHEELS_OFF', 'ELAPSED_TIME', 'AIR_TIME', 'WHEELS_ON', 'TAXI_IN', 'ARRIVAL_TIME', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY',
                   'WEATHER_DELAY', 'CANCELLATION_REASON', 'CANCELLED', 'DIVERTED']

# Função para carregar e salvar cópia processada em parquet se desejável 

def load_flights(input_path, output_path= None):
   df = pd.read_csv(input_path)
   df = remove_cancelled_and_diverted(df)
   df = remove_invalid_scheduled_time(df)
   df = remove_invalid_airport_codes (df)
   df = decompose_scheduled_departure (df)
   df = create_binary_target(df)
   df = drop_unused_columns(df)

   if output_path:
      df.to_parquet(output_path)

   return df
   


# Funções para remover linhas

def remove_cancelled_and_diverted(df):
  return df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]

def remove_invalid_scheduled_time(df):
   return df[df['SCHEDULED_TIME'].notna()]

def remove_invalid_airport_codes(df):
   return df[~df['ORIGIN_AIRPORT'].astype(str).str.isnumeric()]

# Função para decomposição de SCHEDULED_DEPARRTURE

def decompose_scheduled_departure(df):
  df = df.copy()
  df['SCHEDULED_DEPARTURE_HOUR'] =  (df['SCHEDULED_DEPARTURE'] // 100)
  df['SCHEDULED_DEPARTURE_MINUTE'] = (df['SCHEDULED_DEPARTURE'] % 100)
  return df

# Função para criar target binário

def create_binary_target(df):
   df = df.copy()
   df['ARRIVAL_DELAY_CLASS'] = np.where(df['ARRIVAL_DELAY'] >= 15, 1, 0)
   return df

# Função para remover colunas

def drop_unused_columns(df):
   df = df.copy()
   df = df.drop(columns = COLUMNS_TO_DROP, errors ='ignore')
   return df