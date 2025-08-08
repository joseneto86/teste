import pandas as pd
import cudf  # MUDANÇA
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score
from tqdm import tqdm
import cupy as cp  # MUDANÇA: usando cupy em vez de numpy
import numpy as np  # Mantendo para compatibilidade com sklearn
import warnings
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

def get_data_train(window_size, df, features, target):
    X = []
    y = []
    for i in range(window_size, len(df)):
        # Converter para GPU usando cupy
        window = features.iloc[i-window_size:i].values.flatten()
        y.append(target.iloc[i])
        X.append(window)
    return cp.array(X), cp.array(y)

def engenharia_de_dados(df,colunas_para_engenharia):

    for nome_da_feature in colunas_para_engenharia:
        df[f'{nome_da_feature}_MA10'] = df[nome_da_feature].rolling(window=10).mean().fillna(0)
        df[f'{nome_da_feature}_LAG1'] = df[nome_da_feature].shift(1)
        df[f'{nome_da_feature}_STD20'] = df[nome_da_feature].rolling(window=20).std().fillna(0)
        df[f'{nome_da_feature}_ROC5'] = (df[nome_da_feature] - df[nome_da_feature].shift(5)) / df[nome_da_feature].shift(5)

     # ROI protegido contra divisão por zero - versão compatível com cuDF
    roi = df['valorGanho'] / df['valorApostado'].where(df['valorApostado'] > 0, 0)
    df['media_roi_5'] = roi.rolling(window=10).mean().fillna(0)
    df['std_roi_5'] = roi.rolling(window=10).std().fillna(0)

    # Limpeza de inf e NaN restantes
    # Converter para pandas temporariamente para limpeza
    df_pandas = df.to_pandas()
    df_pandas = df_pandas.replace([np.inf, -np.inf], 0)
    df_pandas = df_pandas.fillna(0)
    df = cudf.from_pandas(df_pandas)

    return df

# Carregando os dados
print("🚀 Carregando dados na GPU...")
df_passado = cudf.read_csv('dados_treinamento.csv')
df_futuro = cudf.read_csv('dados_validacao.csv')
print(f"✅ Dados carregados: {len(df_passado)} registros de treinamento, {len(df_futuro)} registros de validação")

colunas_para_engenharia = ['valor', 'diferenca', 'num_apostas', 'valorApostado', 'valorGanho']

print("🔧 Aplicando engenharia de features na GPU...")
df_passado = engenharia_de_dados(df_passado,colunas_para_engenharia)
df_futuro = engenharia_de_dados(df_futuro,colunas_para_engenharia)
print("✅ Engenharia de features concluída")

DADOS_RECENTES_PARA_TREINO = len(df_passado)

# --- CONFIGURAÇÕES DO MODELO ---
DADOS_VALIDACAO = 10
THRESHOLD_DE_DECISAO = 0.70 # NOSSO NOVO PARÂMETRO!
WINDOW_SIZE = 400


# LGBMClassifier configurado para GPU
print("🤖 Configurando LightGBM para GPU...")
model = LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    verbose=-1,
    # Configurações para GPU
    device='CUDA',
    gpu_platform_id=0,
    gpu_device_id=0
)
print("✅ LightGBM configurado para GPU")

# --- SIMULAÇÃO ---
print(f"🎯 Iniciando simulação com {DADOS_VALIDACAO} dados de validação...")
janela_dados_treino = df_passado.copy()
df_teste = df_futuro.head(DADOS_VALIDACAO)
resultados = []
for i in tqdm(range(len(df_teste)), desc="Testando modelo LGBMClassifier com GPU"):
  historico_recente = janela_dados_treino.tail(DADOS_RECENTES_PARA_TREINO)
  features_atuais = historico_recente.select_dtypes(include=[cp.number, np.number]).drop(columns=['result','mes', 'dia_da_semana', 'is_fim_de_semana'], errors='ignore')
  target_atuais = historico_recente['result']
  X_treino, y_treino = get_data_train(WINDOW_SIZE, historico_recente, features_atuais, target_atuais)

  if len(X_treino) > 0:
    # Converter para CPU para o LightGBM (que não suporta cuPy diretamente)
    X_treino_cpu = cp.asnumpy(X_treino)
    y_treino_cpu = cp.asnumpy(y_treino)

    # Treinar modelo
    model.fit(X_treino_cpu, y_treino_cpu)
    ultima_janela = features_atuais.iloc[-WINDOW_SIZE:].values.flatten().reshape(1, -1)

    # --- LÓGICA DE PREVISÃO MODIFICADA ---
    # 1. Obter as probabilidades para ambas as classes: [P(classe_0), P(classe_1)]
    probabilidades = model.predict_proba(ultima_janela)[0]

    # 2. Pegar a probabilidade específica da classe 1
    proba_classe_1 = probabilidades[1]

    # 3. Aplicar nossa regra de negócio com o threshold customizado
    if proba_classe_1 >= THRESHOLD_DE_DECISAO:
        previsao = 1
    else:
        previsao = 0

    # Para o log, continuamos salvando a probabilidade da classe 0, como antes
    proba_log = probabilidades[0]
    # --- FIM DA LÓGICA MODIFICADA ---

  else:
      previsao = 0
      proba_log = 0.5
      
  future_row = df_teste.iloc[i:i+1]
  valor_real = future_row['result'].iloc[0]

  janela_dados_treino = cudf.concat([janela_dados_treino, future_row], ignore_index=True)

  acerto = 1 if previsao == valor_real else 0
  resultados.append({
      'indice': i, 'valor_previsto': previsao, 'valor_real': valor_real, 'acerto': acerto, 'proba': proba_log
  })

# --- AVALIAÇÃO ---
# O restante do código de avaliação permanece o mesmo
resultados_df = pd.DataFrame(resultados)
resultados_df.to_csv('resultados_lgbm_gpu_com_threshold.csv', index=False)
acuracia = accuracy_score(resultados_df['valor_real'], resultados_df['valor_previsto'])
precisao = precision_score(resultados_df['valor_real'], resultados_df['valor_previsto'], zero_division=0)
cm = confusion_matrix(resultados_df['valor_real'], resultados_df['valor_previsto'])

acertos_0 = acertos_1 = erros_0 = erros_1 = 0
if cm.shape == (2, 2):
    acertos_0, erros_1, erros_0, acertos_1 = cm.flatten()
elif len(np.unique(resultados_df['valor_previsto'])) == 1:
    if np.unique(resultados_df['valor_previsto'])[0] == 0:
         if cm.shape[0] > 1:
            acertos_0 = cm[0, 0]
            erros_0 = cm[1, 0]
         else:
            acertos_0 = cm[0, 0]
    else:
        if cm.shape[0] > 1:
            erros_1 = cm[0,0]
            acertos_1 = cm[1,0]
        else:
            erros_1 = cm[0,0]
total = len(resultados_df)

print('\n' + '=' * 50)
print('🚀 RESULTADOS DO TESTE LIGHTGBM COM GPU')
print('=' * 50)
print(f'📊 Configurações:')
print(f'   - Dados de validação: {DADOS_VALIDACAO}')
print(f'   - Window size: {WINDOW_SIZE}')
print(f'   - Threshold: {THRESHOLD_DE_DECISAO}')
try:
    gpu_name = cp.cuda.Device(0).name if cp.cuda.is_available() else "N/A"
except:
    gpu_name = "N/A"
print(f'   - GPU: {gpu_name}')
print('=' * 50)
print('Acertos:')
print(f'- Classe 0 (Não ganhou): {acertos_0} de {total - (acertos_1 + erros_0)}')
print(f'- Classe 1 (Ganhou): {acertos_1} de {acertos_1 + erros_0}')
print('=' * 50)
print('Erros:')
print(f'- Falso Negativo (Previsto 0, Real 1): {erros_0}')
print(f'- Falso Positivo (Previsto 1, Real 0): {erros_1}')
print('=' * 50)
print('\nMétricas Gerais:')
print(f'Acurácia: {acuracia:.4f}')
print(f'Precisão: {precisao:.4f}')
print('=' * 50)
