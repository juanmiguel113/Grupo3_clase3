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
    Crea visualizaciones comprehensivas
    """
    print("\n" + "="*50)
    print("GENERANDO VISUALIZACIONES")
    print("="*50)
    
    # Configurar subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Distribuciones de calificaciones
    subjects = ['math_score', 'reading_score', 'writing_score']
    
    for i, subject in enumerate(subjects, 1):
        plt.subplot(6, 3, i)
        sns.histplot(data=df, x=subject, kde=True, bins=20, alpha=0.7)
        plt.title(f'Distribución de {subject.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        plt.xlabel('Calificación')
        plt.ylabel('Frecuencia')
    
    # 2. Boxplots por género
    for i, subject in enumerate(subjects, 4):
        plt.subplot(6, 3, i)
        sns.boxplot(data=df, x='gender', y=subject)
        plt.title(f'{subject.replace("_", " ").title()} por Género', fontsize=12, fontweight='bold')
        plt.xlabel('Género')
        plt.ylabel('Calificación')
    
    # 3. Impacto socioeconómico
    plt.subplot(6, 3, 7)
    sns.boxplot(data=df, x='lunch', y='average_score')
    plt.title('Rendimiento por Estatus Socioeconómico', fontsize=12, fontweight='bold')
    plt.xlabel('Tipo de Almuerzo')
    plt.ylabel('Promedio de Calificaciones')
    
    # 4. Educación parental
    plt.subplot(6, 3, 8)
    education_order = df.groupby('parental_level_of_education')['average_score'].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x='parental_level_of_education', y='average_score', order=education_order)
    plt.title('Rendimiento por Educación Parental', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Nivel Educativo Parental')
    plt.ylabel('Promedio de Calificaciones')
    
    # 5. Curso de preparación
    plt.subplot(6, 3, 9)
    sns.boxplot(data=df, x='test_preparation_course', y='average_score')
    plt.title('Impacto del Curso de Preparación', fontsize=12, fontweight='bold')
    plt.xlabel('Curso de Preparación')
    plt.ylabel('Promedio de Calificaciones')
    
    # 6. Correlaciones
    plt.subplot(6, 3, 10)
    correlation_matrix = df[subjects + ['average_score']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Matriz de Correlaciones', fontsize=12, fontweight='bold')
    
    # 7. Scatter plot - Matemáticas vs Lectura
    plt.subplot(6, 3, 11)
    sns.scatterplot(data=df, x='math_score', y='reading_score', hue='gender', alpha=0.6)
    plt.title('Matemáticas vs Lectura por Género', fontsize=12, fontweight='bold')
    plt.xlabel('Calificación en Matemáticas')
    plt.ylabel('Calificación en Lectura')
    
    # 8. Violin plot - Rendimiento por grupo étnico
    plt.subplot(6, 3, 12)
    sns.violinplot(data=df, x='race_ethnicity', y='average_score')
    plt.title('Distribución por Grupo Étnico', fontsize=12, fontweight='bold')
    plt.xlabel('Grupo Étnico')
    plt.ylabel('Promedio de Calificaciones')
    
    # 9. Rendimiento por nivel socioeconómico
    plt.subplot(6, 3, 13)
    sns.boxplot(data=df, x='socioeconomic_status', y='average_score')
    plt.title('Rendimiento por Nivel Socioeconómico', fontsize=12, fontweight='bold')
    plt.xlabel('Nivel Socioeconómico')
    plt.ylabel('Promedio de Calificaciones')
    
    # 10. Distribución de nivel de rendimiento
    plt.subplot(6, 3, 14)
    performance_counts = df['performance_level'].value_counts()
    plt.pie(performance_counts.values, labels=performance_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribución de Niveles de Rendimiento', fontsize=12, fontweight='bold')
    
    # 11. Promedio por género y curso de preparación
    plt.subplot(6, 3, 15)
    pivot_data = df.pivot_table(values='average_score', index='gender', columns='test_preparation_course', aggfunc='mean')
    sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', fmt='.1f', cbar_kws={'shrink': 0.8})
    plt.title('Rendimiento: Género vs Preparación', fontsize=12, fontweight='bold')
    plt.xlabel('Curso de Preparación')
    plt.ylabel('Género')
    
    # 12. Análisis de varianza por materia
    plt.subplot(6, 3, 16)
    subject_variance = df[subjects].var()
    subject_variance.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Varianza por Materia', fontsize=12, fontweight='bold')
    plt.xlabel('Materia')
    plt.ylabel('Varianza')
    plt.xticks(rotation=45)
    
    # 13. Distribución conjunta matemáticas-escritura
    plt.subplot(6, 3, 17)
    sns.scatterplot(data=df, x='math_score', y='writing_score', hue='lunch', alpha=0.6)
    plt.title('Matemáticas vs Escritura por Estatus Socioeconómico', fontsize=12, fontweight='bold')
    plt.xlabel('Calificación en Matemáticas')
    plt.ylabel('Calificación en Escritura')
    
    # 14. Comparación de promedios por categorías
    plt.subplot(6, 3, 18)
    category_means = []
    categories = []
    
    for gender in df['gender'].unique():
        for lunch in df['lunch'].unique():
            subset = df[(df['gender'] == gender) & (df['lunch'] == lunch)]
            if len(subset) > 0:
                category_means.append(subset['average_score'].mean())
                categories.append(f"{gender}\n{lunch}")
    
    plt.bar(range(len(category_means)), category_means, color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
    plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
    plt.title('Promedio por Género y Estatus Socioeconómico', fontsize=12, fontweight='bold')
    plt.ylabel('Promedio de Calificaciones')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualizaciones generadas exitosamente")

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
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Importancia de Características en el Modelo', fontsize=14, fontweight='bold')
    plt.xlabel('Importancia')
    plt.ylabel('Características')
    plt.tight_layout()
    plt.show()
    
    # Predicciones vs Valores reales
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_avg, y_pred, alpha=0.6, color='blue')
    plt.plot([y_test_avg.min(), y_test_avg.max()], [y_test_avg.min(), y_test_avg.max()], 'r--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title(f'Predicciones vs Valores Reales (R² = {r2:.3f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return rf_model, feature_importance
