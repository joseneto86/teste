import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from tqdm import tqdm
import itertools
import warnings
from sklearn.model_selection import GridSearchCV
import time

warnings.filterwarnings('ignore')

def get_data_train(window_size, df, features, target):
    """Função para preparar dados de treinamento com janela deslizante"""
    X = []
    y = []
    for i in range(window_size, len(df)):
        window = features.iloc[i-window_size:i].values.flatten()
        y.append(target.iloc[i])
        X.append(window)
    return np.array(X), np.array(y)

def engenharia_de_dados(df, colunas_para_engenharia):
    """Função para engenharia de features"""
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

def avaliar_modelo(model, X_treino, y_treino, X_teste, y_teste):
    """Função para avaliar o modelo e retornar métricas"""
    model.fit(X_treino, y_treino)
    y_pred = model.predict(X_teste)
    y_pred_proba = model.predict_proba(X_teste)
    
    acuracia = accuracy_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred, zero_division=0)
    precisao = precision_score(y_teste, y_pred, zero_division=0)
    
    return {
        'acuracia': acuracia,
        'f1_score': f1,
        'precisao': precisao,
        'probabilidades': y_pred_proba
    }

def simular_backtest(model, df_treino, df_teste, window_size, threshold=0.7):
    """Função para simular backtest com janela deslizante"""
    janela_dados_treino = df_treino.copy()
    resultados = []
    
    for i in range(len(df_teste)):
        historico_recente = janela_dados_treino.tail(len(df_treino))
        
        features_atuais = historico_recente.select_dtypes(include=np.number).drop(
            columns=['result','mes', 'dia_da_semana', 'is_fim_de_semana'], errors='ignore'
        )
        target_atuais = historico_recente['result']
        
        X_treino, y_treino = get_data_train(window_size, historico_recente, features_atuais, target_atuais)
        
        if len(X_treino) > 0:
            model.fit(X_treino, y_treino)
            
            ultima_janela = features_atuais.iloc[-window_size:].values.flatten().reshape(1, -1)
            probabilidades = model.predict_proba(ultima_janela)[0]
            proba_classe_1 = probabilidades[1]
            
            if proba_classe_1 >= threshold:
                previsao = 1
            else:
                previsao = 0
                
            proba_log = probabilidades[0]
        else:
            previsao = 0 
            proba_log = 0.5

        future_row = df_teste.iloc[i:i+1]
        valor_real = future_row['result'].iloc[0]
        
        janela_dados_treino = pd.concat([janela_dados_treino, future_row], ignore_index=True)
        
        acerto = 1 if previsao == valor_real else 0
        resultados.append({
            'indice': i, 
            'valor_previsto': previsao, 
            'valor_real': valor_real, 
            'acerto': acerto, 
            'proba': proba_log
        })

    resultados_df = pd.DataFrame(resultados)
    acuracia = accuracy_score(resultados_df['valor_real'], resultados_df['valor_previsto'])
    f1 = f1_score(resultados_df['valor_real'], resultados_df['valor_previsto'], zero_division=0)
    precisao = precision_score(resultados_df['valor_real'], resultados_df['valor_previsto'], zero_division=0)
    
    return {
        'acuracia': acuracia,
        'f1_score': f1,
        'precisao': precisao,
        'resultados_detalhados': resultados_df
    }

def main():
    print("🚀 INICIANDO TESTE DE HIPERPARÂMETROS LIGHTGBM")
    print("=" * 70)
    
    # Carregando dados
    print("📊 Carregando dados...")
    df_passado = pd.read_csv('dados_treinamento.csv')
    df_futuro = pd.read_csv('dados_validacao.csv')
    
    colunas_para_engenharia = ['valor', 'diferenca', 'num_apostas', 'valorApostado', 'valorGanho']
    
    df_passado = engenharia_de_dados(df_passado, colunas_para_engenharia)
    df_futuro = engenharia_de_dados(df_futuro, colunas_para_engenharia)
    
    # Configurações
    DADOS_VALIDACAO = 25
    THRESHOLD_DE_DECISAO = 0.70
    WINDOW_SIZE = 400  # Usando o melhor window_size encontrado anteriormente
    
    # Definindo hiperparâmetros para testar
    hiperparametros = {
        'n_estimators': [200,300,500],
        'max_depth': [6],
        'learning_rate': [0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_samples': [20],
        'reg_alpha': [0],
        'reg_lambda': [0]
    }
    '''
    hiperparametros = {
        'n_estimators': [200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_samples': [10, 20, 50],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }

    
     LGBMClassifier(
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
    '''
    
    # Gerando todas as combinações de hiperparâmetros
    param_names = list(hiperparametros.keys())
    param_values = list(hiperparametros.values())
    combinacoes = list(itertools.product(*param_values))
    
    print(f"🔍 Testando {len(combinacoes)} combinações de hiperparâmetros")
    print(f"📈 Window size: {WINDOW_SIZE}")
    print(f"🎯 Threshold: {THRESHOLD_DE_DECISAO}")
    print("=" * 70)
    
    # Armazenando resultados
    resultados_hiperparametros = []
    
    # Barra de progresso principal para combinações
    pbar_combinacoes = tqdm(combinacoes, desc="Testando combinações", position=0)
    
    for i, combinacao in enumerate(pbar_combinacoes):
        # Criando dicionário de parâmetros
        params = dict(zip(param_names, combinacao))
        
        # Criando modelo com os parâmetros atuais
        model = LGBMClassifier(
            **params,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            verbose=-1
        )
        
        # Barra de progresso secundária para iterações do backtest
        pbar_iteracoes = tqdm(range(DADOS_VALIDACAO), desc=f"Combinação {i+1}/{len(combinacoes)}", 
                              position=1, leave=False)
        
        # Simulando backtest
        janela_dados_treino = df_passado.copy()
        df_teste = df_futuro.head(DADOS_VALIDACAO)
        resultados = []
        
        for j in pbar_iteracoes:
            historico_recente = janela_dados_treino.tail(len(df_passado))
            
            features_atuais = historico_recente.select_dtypes(include=np.number).drop(
                columns=['result','mes', 'dia_da_semana', 'is_fim_de_semana'], errors='ignore'
            )
            target_atuais = historico_recente['result']
            
            X_treino, y_treino = get_data_train(WINDOW_SIZE, historico_recente, features_atuais, target_atuais)
            
            if len(X_treino) > 0:
                model.fit(X_treino, y_treino)
                
                ultima_janela = features_atuais.iloc[-WINDOW_SIZE:].values.flatten().reshape(1, -1)
                probabilidades = model.predict_proba(ultima_janela)[0]
                proba_classe_1 = probabilidades[1]
                
                if proba_classe_1 >= THRESHOLD_DE_DECISAO:
                    previsao = 1
                else:
                    previsao = 0
                    
                proba_log = probabilidades[0]
            else:
                previsao = 0 
                proba_log = 0.5

            future_row = df_teste.iloc[j:j+1]
            valor_real = future_row['result'].iloc[0]
            
            janela_dados_treino = pd.concat([janela_dados_treino, future_row], ignore_index=True)
            
            acerto = 1 if previsao == valor_real else 0
            resultados.append({
                'indice': j, 
                'valor_previsto': previsao, 
                'valor_real': valor_real, 
                'acerto': acerto, 
                'proba': proba_log
            })
        
        pbar_iteracoes.close()
        
        # Avaliando resultados
        resultados_df = pd.DataFrame(resultados)
        acuracia = accuracy_score(resultados_df['valor_real'], resultados_df['valor_previsto'])
        f1 = f1_score(resultados_df['valor_real'], resultados_df['valor_previsto'], zero_division=0)
        precisao = precision_score(resultados_df['valor_real'], resultados_df['valor_previsto'], zero_division=0)
        
        # Armazenando resultado
        resultado = {
            'combinacao_id': i + 1,
            'acuracia': acuracia,
            'f1_score': f1,
            'precisao': precisao,
            'params': params.copy()
        }
        
        resultados_hiperparametros.append(resultado)
        
        # Atualizando descrição da barra de progresso
        pbar_combinacoes.set_postfix({
            'Precisão': f"{precisao:.4f}",
            'Acurácia': f"{acuracia:.4f}",
            'F1': f"{f1:.4f}"
        })
    
    pbar_combinacoes.close()
    
    # Análise dos resultados
    print("\n" + "=" * 70)
    print("📊 ANÁLISE DOS RESULTADOS")
    print("=" * 70)
    
    # Criando DataFrame com resultados
    df_resultados = pd.DataFrame(resultados_hiperparametros)
    
    # Ordenando por Precisão (métrica principal)
    df_resultados_ordenado = df_resultados.sort_values('precisao', ascending=False)
    
    print("\n🏆 TOP 10 MELHORES COMBINAÇÕES (por Precisão):")
    print("-" * 70)
    
    for i, row in df_resultados_ordenado.head(10).iterrows():
        print(f"#{row['combinacao_id']:3d} | Precisão: {row['precisao']:.4f} | "
              f"Acurácia: {row['acuracia']:.4f} | F1-Score: {row['f1_score']:.4f}")
        print(f"   Parâmetros: {row['params']}")
        print()
    
    # Estatísticas gerais
    print("📈 ESTATÍSTICAS GERAIS:")
    print(f"   Melhor Precisão: {df_resultados['precisao'].max():.4f}")
    print(f"   Média Precisão: {df_resultados['precisao'].mean():.4f}")
    print(f"   Desvio padrão Precisão: {df_resultados['precisao'].std():.4f}")
    print(f"   Melhor Acurácia: {df_resultados['acuracia'].max():.4f}")
    print(f"   Melhor F1-Score: {df_resultados['f1_score'].max():.4f}")
    
    # Encontrando melhor combinação
    melhor_combinacao = df_resultados_ordenado.iloc[0]
    print(f"\n🎯 MELHOR COMBINAÇÃO:")
    print(f"   ID: {melhor_combinacao['combinacao_id']}")
    print(f"   Precisão: {melhor_combinacao['precisao']:.4f}")
    print(f"   Acurácia: {melhor_combinacao['acuracia']:.4f}")
    print(f"   F1-Score: {melhor_combinacao['f1_score']:.4f}")
    print(f"   Parâmetros: {melhor_combinacao['params']}")
    
    # Salvando resultados
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'resultados_hiperparametros_{timestamp}.csv'
    
    # Preparando dados para salvar
    df_salvar = df_resultados.copy()
    df_salvar['params_str'] = df_salvar['params'].astype(str)
    df_salvar = df_salvar.drop('params', axis=1)
    
    df_salvar.to_csv(filename, index=False)
    print(f"\n💾 Resultados salvos em: {filename}")
    
    # Salvando melhor combinação separadamente
    melhor_params = melhor_combinacao['params']
    with open(f'melhor_combinacao_{timestamp}.txt', 'w') as f:
        f.write(f"Melhor Combinação ID: {melhor_combinacao['combinacao_id']}\n")
        f.write(f"Precisão: {melhor_combinacao['precisao']:.4f}\n")
        f.write(f"Acurácia: {melhor_combinacao['acuracia']:.4f}\n")
        f.write(f"F1-Score: {melhor_combinacao['f1_score']:.4f}\n")
        f.write(f"Parâmetros:\n")
        for param, value in melhor_params.items():
            f.write(f"  {param}: {value}\n")
    
    print(f"💾 Melhor combinação salva em: melhor_combinacao_{timestamp}.txt")
    
    # Salvando cada teste de configuração individualmente
    config_filename = f'configuracoes_testadas_{timestamp}.txt'
    with open(config_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CONFIGURAÇÕES TESTADAS - LIGHTGBM HIPERPARÂMETROS\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, row in df_resultados_ordenado.iterrows():
            f.write(f"CONFIGURAÇÃO #{row['combinacao_id']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Precisão: {row['precisao']:.4f}\n")
            f.write(f"Acurácia: {row['acuracia']:.4f}\n")
            f.write(f"F1-Score: {row['f1_score']:.4f}\n")
            f.write("Parâmetros:\n")
            for param, value in row['params'].items():
                f.write(f"  {param}: {value}\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"💾 Todas as configurações salvas em: {config_filename}")
    
    # Gráfico de distribuição (se matplotlib estiver disponível)
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # Distribuição de Precisão
        plt.subplot(2, 2, 1)
        plt.hist(df_resultados['precisao'], bins=20, alpha=0.7, color='red')
        plt.xlabel('Precisão')
        plt.ylabel('Frequência')
        plt.title('Distribuição de Precisão')
        plt.grid(True, alpha=0.3)
        
        # Distribuição de Acurácia
        plt.subplot(2, 2, 2)
        plt.hist(df_resultados['acuracia'], bins=20, alpha=0.7, color='green')
        plt.xlabel('Acurácia')
        plt.ylabel('Frequência')
        plt.title('Distribuição de Acurácia')
        plt.grid(True, alpha=0.3)
        
        # Precisão vs Acurácia
        plt.subplot(2, 2, 3)
        plt.scatter(df_resultados['acuracia'], df_resultados['precisao'], alpha=0.6, color='blue')
        plt.xlabel('Acurácia')
        plt.ylabel('Precisão')
        plt.title('Precisão vs Acurácia')
        plt.grid(True, alpha=0.3)
        
        # Precisão vs F1-Score
        plt.subplot(2, 2, 4)
        plt.scatter(df_resultados['f1_score'], df_resultados['precisao'], alpha=0.6, color='orange')
        plt.xlabel('F1-Score')
        plt.ylabel('Precisão')
        plt.title('Precisão vs F1-Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'grafico_hiperparametros_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: grafico_hiperparametros_{timestamp}.png")
        
    except ImportError:
        print("📊 Matplotlib não disponível - gráfico não gerado")
    
    print("\n" + "=" * 70)
    print("✅ TESTE DE HIPERPARÂMETROS CONCLUÍDO")
    print("=" * 70)

if __name__ == "__main__":
    main()
