import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import warnings
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

def get_data_train(window_size, df, features, target):
    X = []
    y = []
    for i in range(window_size, len(df)):
        window = features.iloc[i-window_size:i].values.flatten()
        y.append(target.iloc[i])
        X.append(window)
    return np.array(X), np.array(y)

def engenharia_de_dados(df,colunas_para_engenharia):
   
    for nome_da_feature in colunas_para_engenharia:
        df[f'{nome_da_feature}_MA10'] = df[nome_da_feature].rolling(window=10).mean().fillna(0)
        df[f'{nome_da_feature}_LAG1'] = df[nome_da_feature].shift(1)
        df[f'{nome_da_feature}_STD20'] = df[nome_da_feature].rolling(window=20).std().fillna(0)
        df[f'{nome_da_feature}_ROC5'] = (df[nome_da_feature] - df[nome_da_feature].shift(5)) / df[nome_da_feature].shift(5)
    
     # ROI protegido contra divisão por zero
    roi = np.where(df['valorApostado'] > 0, df['valorGanho'] / df['valorApostado'], 0)
    roi_series = pd.Series(roi, index=df.index)
    df['media_roi_5'] = roi_series.rolling(window=10).mean().fillna(0)
    df['std_roi_5'] = roi_series.rolling(window=10).std().fillna(0)

    # Limpeza de inf e NaN restantes
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df

# Carregando os dados
df_passado = pd.read_csv('dados_treinamento.csv')
df_futuro = pd.read_csv('dados_validacao.csv')

colunas_para_engenharia = ['valor', 'diferenca', 'num_apostas', 'valorApostado', 'valorGanho']

df_passado = engenharia_de_dados(df_passado,colunas_para_engenharia)
df_futuro = engenharia_de_dados(df_futuro,colunas_para_engenharia)

DADOS_RECENTES_PARA_TREINO = len(df_passado) 

# --- CONFIGURAÇÕES DO MODELO KNN ---
DADOS_VALIDACAO = 100
THRESHOLD_DE_DECISAO = 0.70 # NOSSO NOVO PARÂMETRO!

# Lista de window sizes para testar
window_sizes = [340,380,400]  # 100, 110, 120, ..., 200

print(f"🔍 Testando {len(window_sizes)} tamanhos de janela: {window_sizes}")
print("=" * 60)

# Dicionário para armazenar resultados
resultados_janelas = {}

for window_size in tqdm(window_sizes, desc="Testando janelas"):
    
    print(f"\n📊 Testando window_size = {window_size}")
    
    model = LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    verbose=-1
)

    # --- SIMULAÇÃO ---
    janela_dados_treino = df_passado.copy()
    df_teste = df_futuro.head(DADOS_VALIDACAO)
    resultados = []

    for i in range(len(df_teste)):
        
        historico_recente = janela_dados_treino.tail(DADOS_RECENTES_PARA_TREINO)
        
        features_atuais = historico_recente.select_dtypes(include=np.number).drop(columns=['result','mes', 'dia_da_semana', 'is_fim_de_semana'], errors='ignore')
        target_atuais = historico_recente['result']
        
        X_treino, y_treino = get_data_train(window_size, historico_recente, features_atuais, target_atuais)
        
        if len(X_treino) > 0:
            model.fit(X_treino, y_treino)
        
            ultima_janela = features_atuais.iloc[-window_size:].values.flatten().reshape(1, -1)
            
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
        
        janela_dados_treino = pd.concat([janela_dados_treino, future_row], ignore_index=True)
        
        acerto = 1 if previsao == valor_real else 0
        resultados.append({
            'indice': i, 'valor_previsto': previsao, 'valor_real': valor_real, 'acerto': acerto, 'proba': proba_log
        })

    # --- AVALIAÇÃO ---
    resultados_df = pd.DataFrame(resultados)
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
    
    # Armazenar resultados
    resultados_janelas[window_size] = {
        'acuracia': acuracia,
        'precisao': precisao,
        'acertos_0': acertos_0,
        'acertos_1': acertos_1,
        'erros_0': erros_0,
        'erros_1': erros_1,
        'total': total,
        'f1_score': f1_score(resultados_df['valor_real'], resultados_df['valor_previsto'], zero_division=0)
    }
    
    print(f"   ✅ Acurácia: {acuracia:.4f}")
    print(f"   ✅ Precisão: {precisao:.4f}")
    print(f"   ✅ F1-Score: {resultados_janelas[window_size]['f1_score']:.4f}")

# --- ANÁLISE DOS RESULTADOS ---
print("\n" + "=" * 60)
print("📊 RESULTADOS COMPARATIVOS")
print("=" * 60)

# Criar DataFrame com resultados
df_resultados = pd.DataFrame(resultados_janelas).T
df_resultados.index.name = 'window_size'
df_resultados = df_resultados.reset_index()

# Ordenar por acurácia
df_resultados_ordenado = df_resultados.sort_values('acuracia', ascending=False)

print("\n🏆 TOP 5 MELHORES JANELAS (por acurácia):")
print(df_resultados_ordenado[['window_size', 'acuracia', 'precisao', 'f1_score']].head().to_string(index=False))

print("\n📈 ESTATÍSTICAS GERAIS:")
print(f"   Média de acurácia: {df_resultados['acuracia'].mean():.4f}")
print(f"   Desvio padrão acurácia: {df_resultados['acuracia'].std():.4f}")
print(f"   Melhor acurácia: {df_resultados['acuracia'].max():.4f}")
print(f"   Pior acurácia: {df_resultados['acuracia'].min():.4f}")

# Encontrar melhor janela
melhor_window = df_resultados_ordenado.iloc[0]['window_size']
melhor_acuracia = df_resultados_ordenado.iloc[0]['acuracia']

print(f"\n🎯 MELHOR JANELA: {melhor_window} registros (Acurácia: {melhor_acuracia:.4f})")

# Salvar resultados
df_resultados.to_csv('resultados_janelas_knn.csv', index=False)
print(f"\n💾 Resultados salvos em: resultados_janelas_knn.csv")

# Gráfico de tendência (se matplotlib estiver disponível)
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    
    # Gráfico de acurácia
    plt.subplot(1, 2, 1)
    plt.plot(df_resultados['window_size'], df_resultados['acuracia'], 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Tamanho da Janela')
    plt.ylabel('Acurácia')
    plt.title('Acurácia vs Tamanho da Janela')
    plt.grid(True, alpha=0.3)
    
    # Gráfico de F1-Score
    plt.subplot(1, 2, 2)
    plt.plot(df_resultados['window_size'], df_resultados['f1_score'], 'r-o', linewidth=2, markersize=6)
    plt.xlabel('Tamanho da Janela')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Tamanho da Janela')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grafico_janelas_knn.png', dpi=300, bbox_inches='tight')
    print("📊 Gráfico salvo em: grafico_janelas_knn.png")
    
except ImportError:
    print("📊 Matplotlib não disponível - gráfico não gerado")

print("\n" + "=" * 60)
print("✅ ANÁLISE DE JANELAS CONCLUÍDA")
print("=" * 60) 