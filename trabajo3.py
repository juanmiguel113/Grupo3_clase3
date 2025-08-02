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
    # ===============================================
# 3. VISUALIZACIONES
# ===============================================

def create_visualizations(df):
    """
    Crea visualizaciones comprehensivas con gráficos separados
    """
    print("\n" + "="*50)
    print("GENERANDO VISUALIZACIONES")
    print("="*50)
    
    subjects = ['math_score', 'reading_score', 'writing_score']
    
    # 1. DISTRIBUCIONES DE CALIFICACIONES
    print("\n📊 Generando distribuciones de calificaciones...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Distribución de Calificaciones por Materia', fontsize=16, fontweight='bold')
    
    for i, subject in enumerate(subjects):
        sns.histplot(data=df, x=subject, kde=True, bins=20, alpha=0.7, ax=axes[i])
        axes[i].set_title(f'{subject.replace("_", " ").title()}', fontsize=14)
        axes[i].set_xlabel('Calificación')
        axes[i].set_ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.show()
    
    # 2. ANÁLISIS POR GÉNERO
    print("👥 Generando análisis por género...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Calificaciones por Género', fontsize=16, fontweight='bold')
    
    for i, subject in enumerate(subjects):
        sns.boxplot(data=df, x='gender', y=subject, ax=axes[i])
        axes[i].set_title(f'{subject.replace("_", " ").title()}', fontsize=14)
        axes[i].set_xlabel('Género')
        axes[i].set_ylabel('Calificación')
    
    plt.tight_layout()
    plt.show()
    
    # 3. ANÁLISIS SOCIOECONÓMICO
    print("💰 Generando análisis socioeconómico...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Análisis Socioeconómico', fontsize=16, fontweight='bold')
    
    # Tipo de almuerzo
    sns.boxplot(data=df, x='lunch', y='average_score', ax=axes[0,0])
    axes[0,0].set_title('Rendimiento por Tipo de Almuerzo', fontsize=14)
    axes[0,0].set_xlabel('Tipo de Almuerzo')
    axes[0,0].set_ylabel('Promedio de Calificaciones')
    
    # Educación parental
    education_order = df.groupby('parental_level_of_education')['average_score'].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x='parental_level_of_education', y='average_score', order=education_order, ax=axes[0,1])
    axes[0,1].set_title('Rendimiento por Educación Parental', fontsize=14)
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].set_xlabel('Nivel Educativo Parental')
    axes[0,1].set_ylabel('Promedio de Calificaciones')
    
    # Curso de preparación
    sns.boxplot(data=df, x='test_preparation_course', y='average_score', ax=axes[1,0])
    axes[1,0].set_title('Impacto del Curso de Preparación', fontsize=14)
    axes[1,0].set_xlabel('Curso de Preparación')
    axes[1,0].set_ylabel('Promedio de Calificaciones')
    
    # Nivel socioeconómico creado
    sns.boxplot(data=df, x='socioeconomic_status', y='average_score', ax=axes[1,1])
    axes[1,1].set_title('Rendimiento por Nivel Socioeconómico', fontsize=14)
    axes[1,1].set_xlabel('Nivel Socioeconómico')
    axes[1,1].set_ylabel('Promedio de Calificaciones')
    
    plt.tight_layout()
    plt.show()
    
    # 4. MATRIZ DE CORRELACIONES
    print("🔗 Generando matriz de correlaciones...")
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[subjects + ['average_score']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Matriz de Correlaciones entre Materias', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 5. ANÁLISIS POR GRUPO ÉTNICO
    print("🌍 Generando análisis por grupo étnico...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Análisis por Grupo Étnico', fontsize=16, fontweight='bold')
    
    # Violin plot
    sns.violinplot(data=df, x='race_ethnicity', y='average_score', ax=axes[0])
    axes[0].set_title('Distribución por Grupo Étnico', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_xlabel('Grupo Étnico')
    axes[0].set_ylabel('Promedio de Calificaciones')
    
    # Box plot más detallado
    sns.boxplot(data=df, x='race_ethnicity', y='average_score', ax=axes[1])
    axes[1].set_title('Medidas Estadísticas por Grupo Étnico', fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_xlabel('Grupo Étnico')
    axes[1].set_ylabel('Promedio de Calificaciones')
    
    plt.tight_layout()
    plt.show()
    
    # 6. SCATTER PLOTS - RELACIONES ENTRE MATERIAS
    print("📈 Generando gráficos de dispersión...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Relaciones entre Materias', fontsize=16, fontweight='bold')
    
    # Matemáticas vs Lectura
    sns.scatterplot(data=df, x='math_score', y='reading_score', hue='gender', alpha=0.6, ax=axes[0])
    axes[0].set_title('Matemáticas vs Lectura por Género', fontsize=14)
    axes[0].set_xlabel('Calificación en Matemáticas')
    axes[0].set_ylabel('Calificación en Lectura')
    
    # Matemáticas vs Escritura
    sns.scatterplot(data=df, x='math_score', y='writing_score', hue='lunch', alpha=0.6, ax=axes[1])
    axes[1].set_title('Matemáticas vs Escritura por Estatus Socioeconómico', fontsize=14)
    axes[1].set_xlabel('Calificación en Matemáticas')
    axes[1].set_ylabel('Calificación en Escritura')
    
    # Lectura vs Escritura
    sns.scatterplot(data=df, x='reading_score', y='writing_score', hue='test_preparation_course', alpha=0.6, ax=axes[2])
    axes[2].set_title('Lectura vs Escritura por Preparación', fontsize=14)
    axes[2].set_xlabel('Calificación en Lectura')
    axes[2].set_ylabel('Calificación en Escritura')
    
    plt.tight_layout()
    plt.show()
    
    # 7. ANÁLISIS DE RENDIMIENTO
    print("🎯 Generando análisis de rendimiento...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Análisis de Rendimiento Estudiantil', fontsize=16, fontweight='bold')
    
    # Distribución de niveles de rendimiento
    performance_counts = df['performance_level'].value_counts()
    axes[0,0].pie(performance_counts.values, labels=performance_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0,0].set_title('Distribución de Niveles de Rendimiento', fontsize=14)
    
    # Heatmap: Género vs Preparación
    pivot_data = df.pivot_table(values='average_score', index='gender', columns='test_preparation_course', aggfunc='mean')
    sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', fmt='.1f', ax=axes[0,1])
    axes[0,1].set_title('Rendimiento: Género vs Preparación', fontsize=14)
    
    # Varianza por materia
    subject_variance = df[subjects].var()
    subject_variance.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'], ax=axes[1,0])
    axes[1,0].set_title('Varianza por Materia', fontsize=14)
    axes[1,0].set_xlabel('Materia')
    axes[1,0].set_ylabel('Varianza')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Promedio por categorías combinadas
    category_means = []
    categories = []
    colors = []
    color_map = {'female': 'lightcoral', 'male': 'lightblue'}
    
    for gender in df['gender'].unique():
        for lunch in df['lunch'].unique():
            subset = df[(df['gender'] == gender) & (df['lunch'] == lunch)]
            if len(subset) > 0:
                category_means.append(subset['average_score'].mean())
                categories.append(f"{gender}\n{lunch}")
                colors.append(color_map[gender])
    
    bars = axes[1,1].bar(range(len(category_means)), category_means, color=colors)
    axes[1,1].set_xticks(range(len(categories)))
    axes[1,1].set_xticklabels(categories, rotation=45, ha='right')
    axes[1,1].set_title('Promedio por Género y Estatus Socioeconómico', fontsize=14)
    axes[1,1].set_ylabel('Promedio de Calificaciones')
    
    # Agregar valores en las barras
    for bar, value in zip(bars, category_means):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                      f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualizaciones generadas exitosamente")
    print("📊 Total de gráficos generados: 7 figuras con múltiples visualizaciones")

# ===============================================
# 4. MACHINE LEARNING
# ===============================================

def machine_learning_analysis(df):
    """
    Implementa modelos de Machine Learning para predicción
    """
    print("\n" + "="*50)
    print("ANÁLISIS DE MACHINE LEARNING")
    print("="*50)
    
    if not SKLEARN_AVAILABLE:
        print("⚠️  Sklearn no está disponible. Saltando análisis de ML.")
        return None, None
    
    # Crear dataset para ML3
    ml_df = df.copy()
    
    # Encoding de variables categóricas
    categorical_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 
                       'lunch', 'test_preparation_course']
    
    # Usar un encoder diferente para cada columna
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        ml_df[f'{col}_encoded'] = le.fit_transform(ml_df[col])
        encoders[col] = le
    
    # Variables predictoras
    feature_cols = [f'{col}_encoded' for col in categorical_cols]
    X = ml_df[feature_cols]
    
    # Variables objetivo
    y_math = ml_df['math_score']
    y_reading = ml_df['reading_score']
    y_writing = ml_df['writing_score']
    y_average = ml_df['average_score']
    
    # Dividir datos
    X_train, X_test, y_train_avg, y_test_avg = train_test_split(
        X, y_average, test_size=0.2, random_state=42)
    
    # Entrenar modelo Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train_avg)
    
    # Predicciones
    y_pred = rf_model.predict(X_test)
    
    # Métricas
    rmse = np.sqrt(mean_squared_error(y_test_avg, y_pred))
    r2 = r2_score(y_test_avg, y_pred)
    
    print(f"\n🤖 RESULTADOS DEL MODELO")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {rmse:.2f} puntos")
    print(f"Precisión: {r2*100:.1f}% de la varianza explicada")
    
    # Importancia de características
    feature_importance = pd.DataFrame({
        'feature': categorical_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 IMPORTANCIA DE CARACTERÍSTICAS:")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.3f} ({row['importance']*100:.1f}%)")
    
    # Visualización de importancia
    print("📊 Generando gráfico de importancia de características...")
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Importancia de Características en el Modelo Random Forest', fontsize=16, fontweight='bold')
    plt.xlabel('Importancia Relativa', fontsize=12)
    plt.ylabel('Características', fontsize=12)
    
    # Agregar valores en las barras
    for i, (_, row) in enumerate(feature_importance.iterrows()):
        plt.text(row['importance'] + 0.005, i, f'{row["importance"]:.3f}', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Predicciones vs Valores reales
    print("🎯 Generando gráfico de predicciones vs valores reales...")
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_avg, y_pred, alpha=0.6, color='blue', s=50, edgecolor='white', linewidth=0.5)
    plt.plot([y_test_avg.min(), y_test_avg.max()], [y_test_avg.min(), y_test_avg.max()], 'r--', lw=2, label='Línea perfecta')
    
    plt.xlabel('Valores Reales', fontsize=12)
    plt.ylabel('Predicciones del Modelo', fontsize=12)
    plt.title(f'Predicciones vs Valores Reales\n(R² = {r2:.3f}, RMSE = {rmse:.2f})', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Agregar estadísticas en el gráfico
    plt.text(0.05, 0.95, f'R² Score: {r2:.3f}\nRMSE: {rmse:.2f}\nMuestras: {len(y_test_avg)}', 
             transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return rf_model, feature_importance
