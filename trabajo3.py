import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Importaciones con manejo de errores
try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  Sklearn no está instalado. Las funciones de ML estarán limitadas.")
    print("   Para instalar: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

# Configuración de estilo
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
sns.set_palette("husl")

# ===============================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# ===============================================

def load_and_preprocess_data(file_path):
    """
    Carga y preprocesa el dataset de rendimiento estudiantil
    """
    # Cargar datos
    df = pd.read_csv(file_path)
    
    # Renombrar columnas
    df.columns = [col.lower().replace(" ", "_").replace("/", "_") for col in df.columns]
    
    # Crear variables derivadas
    df['average_score'] = (df['math_score'] + df['reading_score'] + df['writing_score']) / 3
    
    # Categorizar rendimiento
    df['performance_level'] = pd.cut(df['average_score'], 
                                   bins=[0, 60, 80, 100], 
                                   labels=['Bajo', 'Medio', 'Alto'])
    
    # Crear índice socioeconómico
    df['socioeconomic_status'] = df.apply(lambda x: 'Alto' if x['lunch'] == 'standard' and 
                                        x['parental_level_of_education'] in ['bachelor\'s degree', 'master\'s degree'] 
                                        else 'Bajo' if x['lunch'] == 'free/reduced' else 'Medio', axis=1)
    
    print("Dataset cargado exitosamente!")
    print(f"Dimensiones: {df.shape}")
    print(f"Valores nulos: {df.isnull().sum().sum()}")
    print(f"Duplicados: {df.duplicated().sum()}")
    
    return df

# ===============================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS
# ===============================================

def exploratory_analysis(df):
    """
    Realiza análisis exploratorio completo
    """
    print("\n" + "="*50)
    print("ANÁLISIS EXPLORATORIO DE DATOS")
    print("="*50)
    
    # Estadísticas descriptivas
    print("\n📊 ESTADÍSTICAS DESCRIPTIVAS")
    print("-" * 30)
    subjects = ['math_score', 'reading_score', 'writing_score', 'average_score']
    print(df[subjects].describe().round(2))
    
    # Análisis por género
    print("\n👥 ANÁLISIS POR GÉNERO")
    print("-" * 25)
    gender_analysis = df.groupby('gender')[subjects].mean().round(2)
    print(gender_analysis)
    
    # Diferencias de género
    math_diff = gender_analysis.loc['male', 'math_score'] - gender_analysis.loc['female', 'math_score']
    reading_diff = gender_analysis.loc['female', 'reading_score'] - gender_analysis.loc['male', 'reading_score']
    writing_diff = gender_analysis.loc['female', 'writing_score'] - gender_analysis.loc['male', 'writing_score']
    
    print(f"\nDiferencias significativas:")
    print(f"Matemáticas (M > F): {math_diff:.1f} puntos")
    print(f"Lectura (F > M): {reading_diff:.1f} puntos")  
    print(f"Escritura (F > M): {writing_diff:.1f} puntos")
    
    # Análisis socioeconómico
    print("\n💰 ANÁLISIS SOCIOECONÓMICO")
    print("-" * 28)
    lunch_analysis = df.groupby('lunch')[subjects].mean().round(2)
    print(lunch_analysis)
    
    # Impacto socioeconómico
    socio_impact = lunch_analysis.loc['standard', 'average_score'] - lunch_analysis.loc['free/reduced', 'average_score']
    print(f"\nImpacto socioeconómico: {socio_impact:.1f} puntos de diferencia")
    
    # Análisis por educación parental
    print("\n🎓 EDUCACIÓN PARENTAL")
    print("-" * 20)
    education_analysis = df.groupby('parental_level_of_education')['average_score'].mean().sort_values(ascending=False)
    print(education_analysis.round(2))
    
    # Curso de preparación
    print("\n📚 CURSO DE PREPARACIÓN")
    print("-" * 22)
    prep_analysis = df.groupby('test_preparation_course')[subjects].mean().round(2)
    print(prep_analysis)
    
    prep_impact = prep_analysis.loc['completed', 'average_score'] - prep_analysis.loc['none', 'average_score']
    print(f"\nImpacto del curso: {prep_impact:.1f} puntos de mejora")