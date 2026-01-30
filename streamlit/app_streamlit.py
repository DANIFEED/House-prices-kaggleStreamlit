import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
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

EXPECTED_COLUMNS = [
    'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
    'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
    'PoolArea', 'MiscVal', 'MoSold', 'YrSold',
    
    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
    'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
    'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
    'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
    'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
    'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
    'MiscFeature', 'SaleType', 'SaleCondition'
]

FEATURES_TO_DROP = [
    'Utilities', 'Street', 'PoolArea', 'PoolQC', 'Condition2', 
    'RoofMatl', 'Heating', 'LowQualFinSF', '3SsnPorch', 
    'MiscFeature', 'MiscVal', 'Alley', 'BsmtHalfBath', 
    'BsmtFinSF2', 'LandSlope', 'LandContour', 'YrSold', 'MoSold'
]

class HousePricesSmartImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stats_ = {}
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = X.columns.tolist()
        self.feature_names_out_ = X.columns.tolist()
        self.stats_ = {}
        
        if 'LotFrontage' in X.columns and 'Neighborhood' in X.columns:
            self.stats_['lot_medians'] = X.groupby('Neighborhood')['LotFrontage'].median().to_dict()
            self.stats_['lot_overall'] = X['LotFrontage'].median()

        if 'MSZoning' in X.columns and 'MSSubClass' in X.columns:
            mode_series = X.groupby('MSSubClass')['MSZoning'].agg(
                lambda x: x.mode().iat[0] if not x.mode().empty else np.nan
            )
            self.stats_['zoning_modes'] = mode_series.to_dict()
            
            overall_zoning_mode = X['MSZoning'].mode()
            self.stats_['zoning_overall'] = overall_zoning_mode.iat[0] if not overall_zoning_mode.empty else None
            
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        if 'LotFrontage' in X.columns:
            X['LotFrontage'] = X['LotFrontage'].fillna(X['Neighborhood'].map(self.stats_.get('lot_medians', {})))
            X['LotFrontage'] = X['LotFrontage'].fillna(self.stats_.get('lot_overall'))

        if 'MSZoning' in X.columns:
            X['MSZoning'] = X['MSZoning'].fillna(X['MSSubClass'].map(self.stats_.get('zoning_modes', {})))
            X['MSZoning'] = X['MSZoning'].fillna(self.stats_.get('zoning_overall'))

        if 'GarageYrBlt' in X.columns and 'YearBuilt' in X.columns:
            X['GarageYrBlt'] = X['GarageYrBlt'].fillna(X['YearBuilt'])

        if 'Functional' in X.columns:
            X['Functional'] = X['Functional'].fillna('Typ')

        return X.values

    def get_feature_names_out(self, input_features=None):
        """Возвращает имена признаков после преобразования"""
        if input_features is not None:
            return input_features
        elif self.feature_names_out_ is not None:
            return self.feature_names_out_
        else:
            raise ValueError("Feature names not available")

@st.cache_resource
def load_model():
    """Загружает модель с обработкой разных форматов"""
    model_file = "ml_ensemble_rfr_lgbm_catb.pkl"
    
    if not os.path.exists(model_file):
        return None, f"❌ Файл '{model_file}' не найден. Убедитесь, что модель в той же папке."
    
    try:
        model = joblib.load(model_file)
        return model, "✅ Модель загружена успешно"
    except:
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            return model, "✅ Модель загружена успешно"
        except:
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f, encoding='latin1')
                return model, "✅ Модель загружена"
            except Exception as e:
                return None, f"❌ Ошибка загрузки модели: {str(e)[:100]}"

def preprocess_simple(df):
    """Упрощенная обработка данных для House Prices"""
    df = df.copy()
    
    if 'Id' in df.columns:
        ids = df['Id'].copy()
        df = df.drop('Id', axis=1)
    else:
        ids = pd.Series(range(1, len(df) + 1), name='Id')
    
    if 'SalePrice' in df.columns:
        df = df.drop('SalePrice', axis=1)
    
    try:
        cols_to_drop = [col for col in FEATURES_TO_DROP if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            st.info(f"ℹ️ Удалены колонки: {', '.join(cols_to_drop)}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if 'Functional' in df.columns:
            df['Functional'] = df['Functional'].fillna('Typ')
        
        if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
            df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
        
        cols_fillna_0 = [
            'MasVnrArea', 'BsmtFullBath', 'BsmtFinSF1', 'BsmtUnfSF', 
            'TotalBsmtSF', 'GarageCars', 'GarageArea'
        ]
        
        cols_fillna_none = [
            'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 
            'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 
            'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
        ]
        
        cols_fillna_mode = [
            'MSZoning', 'SaleType', 'KitchenQual', 'Electrical', 
            'Exterior1st', 'Exterior2nd'
        ]
        
        for col in cols_fillna_0:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        for col in cols_fillna_none:
            if col in df.columns:
                df[col] = df[col].fillna('None')
        
        for col in cols_fillna_mode:
            if col in df.columns:
                if df[col].notna().any():
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_value)
                else:
                    df[col] = df[col].fillna('Unknown')
        
        for col in categorical_cols:
            if col in df.columns and df[col].isna().any():
                if col not in cols_fillna_none + cols_fillna_mode and col != 'Functional':
                    df[col] = df[col].fillna('Unknown')
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(df[categorical_cols])
        
            encoded_feature_names = []
            for i, col in enumerate(categorical_cols):
                categories = encoder.categories_[i]
                for category in categories:
                    encoded_feature_names.append(f"{col}_{category}")
            
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=df.index)
    
            numeric_df = df.select_dtypes(include=[np.number])
            df_processed = pd.concat([numeric_df, encoded_df], axis=1)
        else:
            df_processed = df
    
        base_features = [
            'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
            'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF',
            'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
            'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
            'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch',
            'MiscVal'
        ]
        
        all_expected_features = []
        
        for feature in base_features:
            all_expected_features.append(feature)
        
        categorical_prefixes = ['MSZoning_', 'Neighborhood_', 'HouseStyle_', 'RoofStyle_', 
                               'Exterior1st_', 'Exterior2nd_', 'Foundation_', 'HeatingQC_',
                               'CentralAir_', 'KitchenQual_', 'Functional_', 'GarageType_',
                               'SaleType_', 'SaleCondition_']
        
        for prefix in categorical_prefixes:
            for suffix in ['Typical', 'Average', 'Good', 'Excellent', 'Fair', 'Poor', 
                          'Y', 'N', 'WD', 'New', 'COD', 'CWD', 'Con', 'ConLw', 'ConLI',
                          'ConLD', 'Oth', 'Normal', 'Abnorml', 'Partial', 'Family',
                          'Alloca', 'AdjLand']:
                all_expected_features.append(f"{prefix}{suffix}")
        
        for feature in all_expected_features:
            if feature not in df_processed.columns:
                df_processed[feature] = 0
        
        if len(df_processed.columns) > 109:
            df_processed = df_processed.iloc[:, :109]
        
        scaler = StandardScaler()
        processed_data = scaler.fit_transform(df_processed)
        
        st.success(f"✅ Данные обработаны (упрощенный режим): {processed_data.shape[0]} строк, {processed_data.shape[1]} признаков")
        return processed_data, ids
        
    except Exception as e:
        st.error(f"❌ Ошибка при простой обработке: {str(e)}")
        return None, None

@st.cache_resource
def create_and_fit_pipeline(_X_train, _y_train):
    """Создает и обучает полный пайплайн обработки"""

    if 'Id' in _X_train.columns:
        _X_train = _X_train.drop('Id', axis=1)
    
    features_to_drop = FEATURES_TO_DROP
    
    cols_fillna_0 = [
        'MasVnrArea', 'BsmtFullBath', 'BsmtFinSF1', 'BsmtUnfSF', 
        'TotalBsmtSF', 'GarageCars', 'GarageArea'
    ]
    
    cols_fillna_none = [
        'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 
        'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
    ]
    
    cols_fillna_mode = [
        'MSZoning', 'SaleType', 'KitchenQual', 'Electrical', 
        'Exterior1st', 'Exterior2nd'
    ]
    
    imputer = ColumnTransformer([
        ('drop_features', 'drop', features_to_drop),
        ('smart_imputer', HousePricesSmartImputer(), 
         ['LotFrontage', 'Neighborhood', 'GarageYrBlt', 'YearBuilt', 'Functional']),
        ('zero', SimpleImputer(strategy='constant', fill_value=0), cols_fillna_0),
        ('none', SimpleImputer(strategy='constant', fill_value='None'), cols_fillna_none),
        ('mode', SimpleImputer(strategy='most_frequent'), cols_fillna_mode)
    ], remainder='passthrough', verbose_feature_names_out=False)
    
    X_imputed = imputer.fit_transform(_X_train)
    feature_names = imputer.get_feature_names_out()
    X_imputed_df = pd.DataFrame(X_imputed, columns=feature_names, index=_X_train.index)
    
    cat_cols = X_imputed_df.select_dtypes(include=['object']).columns.tolist()
    
    split = 4
    col_cat_ohe = [col for col in cat_cols if X_imputed_df[col].nunique() <= split]
    
    col_num = X_imputed_df.select_dtypes(include=['number']).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('one_hot_encoding', OneHotEncoder(handle_unknown='ignore', sparse_output=False), col_cat_ohe),
            ('standard_scaler', StandardScaler(), col_num)
        ],
        verbose_feature_names_out=False,
        remainder='drop'
    )
    
    X_processed = preprocessor.fit_transform(X_imputed_df)
    
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names_processed = preprocessor.get_feature_names_out()
    else:
        feature_names_processed = []
        for name, trans, cols in preprocessor.transformers_:
            if trans == 'drop':
                continue
            if hasattr(trans, 'get_feature_names_out'):
                feature_names_processed.extend(trans.get_feature_names_out(cols))
            else:
                feature_names_processed.extend(cols)
    
    if len(feature_names_processed) > 109:
        feature_names_processed = feature_names_processed[:109]
        X_processed = X_processed[:, :109]
    
    return {
        'imputer': imputer,
        'preprocessor': preprocessor,
        'feature_names': feature_names_processed,
        'col_cat_ohe': col_cat_ohe,
        'col_num': col_num
    }

def preprocess_with_pipeline(df, pipeline):
    """Обрабатывает новые данные обученным пайплайном"""
    df = df.copy()
    
    if 'Id' in df.columns:
        ids = df['Id'].copy()
        df_for_processing = df.drop('Id', axis=1)
    else:
        ids = pd.Series(range(1, len(df) + 1), name='Id')
        df_for_processing = df.copy()
    
    if 'SalePrice' in df_for_processing.columns:
        df_for_processing = df_for_processing.drop('SalePrice', axis=1)
    
    try:
        progress_bar = st.progress(0)
        
        progress_bar.progress(20)
        with st.spinner("🔧 Шаг 1: Обработка пропущенных значений..."):
            df_imputed = pipeline['imputer'].transform(df_for_processing)
            feature_names = pipeline['imputer'].get_feature_names_out()
            df_imputed = pd.DataFrame(df_imputed, columns=feature_names, index=df_for_processing.index)
        
        progress_bar.progress(50)
        with st.spinner("🔧 Шаг 2: Кодирование и масштабирование признаков..."):
            df_processed = pipeline['preprocessor'].transform(df_imputed)
        
        if df_processed.shape[1] > 109:
            df_processed = df_processed[:, :109]
        elif df_processed.shape[1] < 109:
            zeros_to_add = 109 - df_processed.shape[1]
            df_processed = np.hstack([df_processed, np.zeros((df_processed.shape[0], zeros_to_add))])
        
        progress_bar.progress(100)
        st.success(f"✅ Данные полностью обработаны: {df_processed.shape[0]} строк, {df_processed.shape[1]} признаков")
        return df_processed, ids
        
    except Exception as e:
        st.error(f"❌ Ошибка при обработке данных: {str(e)}")
        return None, None

st.markdown("<h1 class='main-header'>🏡 Прогноз стоимости недвижимости</h1>", unsafe_allow_html=True)

model, model_message = load_model()

if model is None:
    st.error(model_message)
    st.stop()

st.success(model_message)

st.markdown("<h2 class='section-header'>📚 Загрузите тренировочные данные</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='upload-box'>
    Для наилучших результатов загрузите тренировочный CSV файл (с колонкой 'SalePrice').<br>
    Это позволит создать точный пайплайн обработки, как при обучении модели.
</div>
""", unsafe_allow_html=True)

uploaded_train_file = st.file_uploader(
    "Выберите тренировочный CSV файл",
    type=['csv'],
    key="train_uploader"
)

if uploaded_train_file is not None:
    with st.spinner("📥 Загружаю тренировочные данные..."):
        try:
            train_df = pd.read_csv(uploaded_train_file)
            
            if 'SalePrice' not in train_df.columns:
                st.error("❌ В тренировочных данных нет колонки 'SalePrice'")
            else:
                st.success(f"✅ Тренировочные данные загружены: {train_df.shape[0]} строк, {train_df.shape[1]} колонок")
                
                X_train = train_df.drop('SalePrice', axis=1)
                y_train = train_df['SalePrice']
                
                with st.spinner("🎯 Создаю и обучаю пайплайн обработки..."):
                    try:
                        pipeline = create_and_fit_pipeline(X_train, y_train)
                        
                        st.session_state['pipeline'] = pipeline
                        st.session_state['pipeline_ready'] = True
                        st.session_state['train_size'] = len(train_df)
                        
                        st.success(f"✅ Пайплайн успешно создан и обучен на {len(train_df)} примерах!")
                        
                        with st.expander("📊 Информация о пайплайне"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("OHE признаков", len(pipeline['col_cat_ohe']))
                            with col2:
                                st.metric("Числовых признаков", len(pipeline['col_num']))
                            with st.container():
                                st.write(f"**Всего признаков после обработки:** {len(pipeline['feature_names'])}")
                                st.write(f"**Модель ожидает:** 109 признаков")
                            
                    except Exception as e:
                        st.error(f"❌ Ошибка при создании пайплайна: {str(e)}")
                        st.info("ℹ️ Будет использована упрощенная обработка данных")
                        st.session_state['pipeline_ready'] = False
                        
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке файла: {e}")

if 'pipeline_ready' not in st.session_state or not st.session_state['pipeline_ready']:
    st.markdown("""
    <div class='warning-box'>
    <strong>⚠️ Внимание:</strong> Пайплайн обработки не создан. Для наилучших результатов загрузите тренировочные данные.<br>
    Будет использована упрощенная обработка данных.
    </div>
    """, unsafe_allow_html=True)
    
    st.session_state['use_simple_processing'] = True

st.markdown("<h2 class='section-header'>📁 Загрузите CSV файл для предсказания</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='upload-box'>
    Загрузите CSV файл с данными о недвижимости для предсказания.<br>
    <small>Поддерживаются стандартные признаки из набора данных House Prices</small>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Выберите CSV файл для предсказания",
    type=['csv'],
    key="predict_uploader"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Файл загружен: {uploaded_file.name}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Строк", df.shape[0])
        with col2:
            st.metric("Столбцов", df.shape[1])
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Числовых колонок", len(numeric_cols))
        
        with st.expander("👁️ Посмотреть данные"):
            st.dataframe(df.head())
        
        if st.button("🎯 Предсказать стоимость для всех объектов", type="primary", use_container_width=True):
            with st.spinner("Обрабатываю данные и делаю предсказание..."):
                try:
                    if ('pipeline_ready' in st.session_state and st.session_state['pipeline_ready'] and 
                        'use_simple_processing' not in st.session_state):
                        
                        st.info("🔧 Используется полный пайплайн обработки")
                        pipeline = st.session_state['pipeline']
                        processed_data, ids = preprocess_with_pipeline(df, pipeline)
                    else:
                        
                        st.info("🔧 Используется упрощенная обработка данных")
                        processed_data, ids = preprocess_simple(df)
                    
                    if processed_data is None:
                        st.error("❌ Не удалось обработать данные")
                        st.stop()
                    
                    st.write(f"📊 Количество признаков после обработки: {processed_data.shape[1]}")
                    
                    predictions = model.predict(processed_data)
                    
                    try:
                        predictions = np.expm1(predictions)
                        st.info("ℹ️ Применено преобразование: expm1 (log1p → доллары)")
                    except:
                        st.info("ℹ️ Предсказания уже в долларах")
                    
                    results_df = pd.DataFrame({
                        'Id': ids,
                        'Predicted_Price': predictions
                    })
                    
                    st.markdown("<h2 class='section-header'>📈 Результаты предсказания</h2>", unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Объектов", len(predictions))
                    col2.metric("Средняя цена", f"${predictions.mean():,.0f}")
                    col3.metric("Минимальная", f"${predictions.min():,.0f}")
                    col4.metric("Максимальная", f"${predictions.max():,.0f}")
                    
                    if len(predictions) > 1:
                        st.subheader("📊 Распределение цен")
                        st.bar_chart(results_df.set_index('Id')['Predicted_Price'])
                        
                        if predictions.std() < 50000:
                            st.warning(f"⚠️ Цены слишком однородные (стандартное отклонение: ${predictions.std():,.0f})")
                    
                    st.subheader("📋 Детализация предсказаний")
                    st.dataframe(
                        results_df.style.format({'Predicted_Price': '${:,.0f}'}),
                        height=400,
                        use_container_width=True
                    )
                    
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
                    st.info("ℹ️ Возможно, данные имеют неверный формат или отсутствуют необходимые признаки")
                    
    except Exception as e:
        st.error(f"❌ Ошибка при чтении файла: {e}")

