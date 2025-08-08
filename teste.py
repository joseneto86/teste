import cudf
import cuml

print(f"Versão do cuDF: {cudf.__version__}")
print(f"Versão do cuML: {cuml.__version__}")

# Teste do cuDF: Criar um DataFrame na GPU
try:
    gdf = cudf.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
    print("\nDataFrame do cuDF criado na GPU com sucesso:")
    print(gdf)
    print("\nTeste do cuDF: OK")
except Exception as e:
    print(f"\nErro ao criar DataFrame do cuDF: {e}")

# Teste do cuML: Verificar se um modelo pode ser instanciado
try:
    from cuml.cluster import DBSCAN
    dbscan = DBSCAN(eps=1.0, min_samples=2)
    print("\nModelo DBSCAN do cuML instanciado com sucesso.")
    print("Teste do cuML: OK")
except Exception as e:
    print(f"\nErro ao instanciar modelo do cuML: {e}")