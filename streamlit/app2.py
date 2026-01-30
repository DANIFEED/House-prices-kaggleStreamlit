import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2D3748;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4A5568;
        margin-top: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E2E8F0;
    }
    
    .upload-box {
        border: 2px dashed #CBD5E0;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .step-box {
        background-color: #F7FAFC;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4299E1;
    }
    
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Сначала покажем дебаг информацию
st.sidebar.write("🔧 Дебаг информация")

@st.cache_resource
def load_model():
    """Загружает модель с обработкой разных форматов"""
    # Пробуем разные имена файлов
    possible_files = [
        "ml_baseline_rfr.pkl",
        "./ml_baseline_rfr.pkl",
        "/mount/src/your-repo-name/ml_baseline_rfr.pkl",  # путь на Streamlit Cloud
    ]
    
    for model_file in possible_files:
        try:
            st.sidebar.write(f"Пробую загрузить: {model_file}")
            
            if os.path.exists(model_file):
                st.sidebar.success(f"✅ Файл найден: {model_file}")
                st.sidebar.write(f"Размер файла: {os.path.getsize(model_file) / 1024 / 1024:.2f} MB")
                
                try:
                    # Сначала пробуем joblib
                    model = joblib.load(model_file)
                    st.sidebar.success("✅ Модель загружена через joblib")
                    return model, "✅ Модель загружена успешно"
                except Exception as e:
                    st.sidebar.warning(f"Joblib не сработал: {str(e)[:100]}")
                    
                    try:
                        # Пробуем pickle
                        with open(model_file, 'rb') as f:
                            model = pickle.load(f)
                        st.sidebar.success("✅ Модель загружена через pickle")
                        return model, "✅ Модель загружена успешно"
                    except Exception as e2:
                        st.sidebar.warning(f"Pickle не сработал: {str(e2)[:100]}")
                        
                        try:
                            # Пробуем pickle с latin1
                            with open(model_file, 'rb') as f:
                                model = pickle.load(f, encoding='latin1')
                            st.sidebar.success("✅ Модель загружена через pickle (latin1)")
                            return model, "✅ Модель загружена успешно"
                        except Exception as e3:
                            st.sidebar.error(f"Все методы не сработали: {str(e3)[:100]}")
            else:
                st.sidebar.info(f"Файл не найден: {model_file}")
                
        except Exception as e:
            st.sidebar.error(f"Ошибка при проверке {model_file}: {str(e)[:100]}")
    
    # Покажем список файлов в директории
    st.sidebar.write("📁 Содержимое текущей директории:")
    try:
        files = os.listdir('.')
        for file in files:
            st.sidebar.write(f"  - {file}")
    except Exception as e:
        st.sidebar.error(f"Не могу прочитать директорию: {e}")
    
    return None, "❌ Файл модели не найден. Убедитесь, что файл 'ml_baseline_rfr.pkl' находится в корневой папке репозитория."

def preprocess_data_simple(df):
    """Упрощенная обработка данных"""
    df = df.copy()
    
    # Сохраняем Id
    if 'Id' in df.columns:
        ids = df['Id'].copy()
        df = df.drop('Id', axis=1)
    else:
        ids = pd.Series(range(1, len(df) + 1), name='Id')
    
    # Удаляем SalePrice если есть
    if 'SalePrice' in df.columns:
        df = df.drop('SalePrice', axis=1)
    
    try:
        # Простая обработка
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Заполняем пропуски
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 0)
        
        for col in categorical_cols:
            if df[col].isnull().any():
                if df[col].notna().any():
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
                else:
                    df[col] = df[col].fillna('Unknown')
        
        # Кодируем категориальные признаки
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        
        if len(categorical_cols) > 0:
            # Используем простой one-hot для основных признаков
            important_cats = ['MSZoning', 'Neighborhood', 'HouseStyle', 'KitchenQual', 
                             'SaleType', 'SaleCondition', 'CentralAir']
            
            encoded_dfs = []
            for col in important_cats:
                if col in df.columns:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    encoded_dfs.append(dummies)
            
            # Удаляем оригинальные категориальные колонки
            df = df.drop(categorical_cols, axis=1, errors='ignore')
            
            # Добавляем закодированные
            if encoded_dfs:
                df = pd.concat([df] + encoded_dfs, axis=1)
        
        # Обеспечиваем нужное количество признаков (109)
        current_features = df.shape[1]
        
        if current_features > 109:
            df = df.iloc[:, :109]
            st.info(f"ℹ️ Уменьшено количество признаков с {current_features} до 109")
        elif current_features < 109:
            missing_features = 109 - current_features
            for i in range(missing_features):
                df[f'feature_{i}'] = 0
            st.info(f"ℹ️ Добавлено {missing_features} пустых признаков")
        
        # Масштабирование
        scaler = StandardScaler()
        processed_data = scaler.fit_transform(df)
        
        st.success(f"✅ Данные обработаны: {processed_data.shape[0]} строк, {processed_data.shape[1]} признаков")
        return processed_data, ids
        
    except Exception as e:
        st.error(f"❌ Ошибка при обработке данных: {str(e)}")
        return None, None

# Основная часть приложения
st.markdown("<h1 class='main-header'>🏡 Прогноз стоимости недвижимости</h1>", unsafe_allow_html=True)

st.info("""
Это приложение использует модель машинного обучения для прогнозирования стоимости жилой недвижимости.
Загрузите CSV файл с данными для получения предсказаний.
""")

# Загружаем модель
model, model_message = load_model()

if model is None:
    st.error(model_message)
    
    # Дополнительная помощь
    with st.expander("🛠️ Помощь по устранению проблемы"):
        st.write("""
        **Проблема:** Файл модели не найден на Streamlit Cloud.
        
        **Решение:**
        1. Убедитесь, что файл `ml_baseline_rfr.pkl` загружен в ваш репозиторий
        2. Проверьте, что файл находится в корневой папке (не в подпапке)
        3. Убедитесь, что размер файла < 200MB
        4. Перезагрузите репозиторий на Streamlit Cloud
        
        **Структура должна быть:**
        ```
        ваш-репозиторий/
        ├── app.py
        ├── ml_baseline_rfr.pkl
        └── requirements.txt
        ```
        """)
    st.stop()

st.success(model_message)

# Информация о модели
with st.expander("📊 Информация о модели"):
    st.write(f"**Тип модели:** {type(model).__name__}")
    
    if hasattr(model, 'n_features_in_'):
        st.write(f"**Ожидаемое количество признаков:** {model.n_features_in_}")
    elif hasattr(model, 'feature_importances_'):
        st.write(f"**Количество признаков:** {len(model.feature_importances_)}")
    
    if hasattr(model, 'n_estimators'):
        st.write(f"**Количество деревьев:** {model.n_estimators}")

# Секция для загрузки данных
st.markdown("<h2 class='section-header'>📁 Загрузите данные для предсказания</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='upload-box'>
    <h4>📋 Требования к данным</h4>
    <p>Загрузите CSV файл с данными о недвижимости</p>
    <p><small>Файл должен содержать стандартные признаки House Prices dataset</small></p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Выберите CSV файл",
    type=['csv'],
    key="predict_uploader"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Файл загружен: {uploaded_file.name}")
        
        # Основная информация
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Строк", df.shape[0])
        with col2:
            st.metric("Столбцов", df.shape[1])
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Числовых признаков", len(numeric_cols))
        
        # Предпросмотр
        with st.expander("👁️ Предпросмотр данных"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Кнопка предсказания
        if st.button("🎯 Предсказать стоимость", type="primary", use_container_width=True):
            with st.spinner("🔄 Обрабатываю данные..."):
                processed_data, ids = preprocess_data_simple(df)
                
                if processed_data is None:
                    st.error("❌ Не удалось обработать данные")
                    st.stop()
                
                st.write(f"📊 Обработано признаков: {processed_data.shape[1]}")
                
                with st.spinner("🤖 Делаю предсказания..."):
                    try:
                        predictions = model.predict(processed_data)
                        
                        # Преобразование из логарифмированной шкалы
                        try:
                            predictions = np.expm1(predictions)
                            st.info("ℹ️ Применено преобразование из логарифмической шкалы")
                        except:
                            st.info("ℹ️ Предсказания в исходной шкале")
                        
                        # Создаем результаты
                        results_df = pd.DataFrame({
                            'Id': ids,
                            'Predicted_Price': predictions.round(2)
                        })
                        
                        # Статистика
                        st.markdown("<h2 class='section-header'>📈 Результаты</h2>", unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Объектов", len(predictions))
                        with col2:
                            st.metric("Средняя цена", f"${predictions.mean():,.0f}")
                        with col3:
                            st.metric("Минимальная", f"${predictions.min():,.0f}")
                        with col4:
                            st.metric("Максимальная", f"${predictions.max():,.0f}")
                        
                        # Визуализация
                        if len(predictions) > 1:
                            st.subheader("📊 Распределение цен")
                            st.bar_chart(results_df.set_index('Id')['Predicted_Price'])
                        
                        # Таблица с результатами
                        st.subheader("📋 Детализация предсказаний")
                        st.dataframe(
                            results_df.style.format({'Predicted_Price': '${:,.0f}'}),
                            height=400,
                            use_container_width=True
                        )
                        
                        # Кнопка для скачивания
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Скачать предсказания (CSV)",
                            data=csv,
                            file_name=f"house_price_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            type="primary"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка при предсказании: {str(e)}")
                        st.info("Попробуйте другой файл или проверьте формат данных")
                
    except Exception as e:
        st.error(f"❌ Ошибка при чтении файла: {e}")

# Футер
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; font-size: 0.9rem;">
    <p><strong>House Price Prediction Tool</strong></p>
    <p>Используется модель Random Forest для прогнозирования стоимости недвижимости</p>
</div>
""", unsafe_allow_html=True)