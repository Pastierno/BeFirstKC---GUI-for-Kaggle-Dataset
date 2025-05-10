import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QComboBox, QScrollArea, QGroupBox, QLabel, QMessageBox,
                             QListWidget, QSplitter)
from PyQt5.QtCore import Qt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import seaborn as sns
import io

class DataFrameViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.train_df = None
        self.test_df = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Data Analysis Tool')
        self.setGeometry(100, 100, 1200, 800)
        
        self.tabs = QTabWidget()
        self.create_file_tab()
        self.create_preprocess_tab()
        self.create_visualization_tab()
        self.create_model_tab()
        
        self.setCentralWidget(self.tabs)
        self.show()
    
    def create_file_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        self.train_btn = QPushButton('Load Train Dataset')
        self.train_btn.clicked.connect(lambda: self.load_file('train'))
        self.test_btn = QPushButton('Load Test Dataset')
        self.test_btn.clicked.connect(lambda: self.load_file('test'))
        
        layout.addWidget(self.train_btn)
        layout.addWidget(self.test_btn)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Load Data")
    
    def create_preprocess_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout()
        
        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['Train', 'Test'])
        self.dataset_combo.currentIndexChanged.connect(self.update_preprocess_info)
        
        # Info display
        info_group = QGroupBox('Dataset Info')
        info_layout = QVBoxLayout()
        self.info_text = QLabel()
        self.info_text.setWordWrap(True)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        
        # NaN display
        nan_group = QGroupBox('Missing Values')
        nan_layout = QVBoxLayout()
        self.nan_text = QLabel()
        self.nan_text.setWordWrap(True)
        nan_layout.addWidget(self.nan_text)
        nan_group.setLayout(nan_layout)
        
        # Preprocessing buttons
        btn_layout = QHBoxLayout()
        self.drop_nan_btn = QPushButton('Drop NaN Values')
        self.drop_nan_btn.clicked.connect(self.drop_nan)
        
        self.impute_group = QGroupBox('Impute NaN')
        impute_layout = QHBoxLayout()
        self.impute_combo = QComboBox()
        self.impute_combo.addItems(['Mean', 'Median', 'Mode'])
        self.impute_btn = QPushButton('Impute')
        self.impute_btn.clicked.connect(self.impute_nan)
        impute_layout.addWidget(self.impute_combo)
        impute_layout.addWidget(self.impute_btn)
        self.impute_group.setLayout(impute_layout)
        
        btn_layout.addWidget(self.drop_nan_btn)
        btn_layout.addWidget(self.impute_group)
        
        main_layout.addWidget(self.dataset_combo)
        main_layout.addWidget(info_group)
        main_layout.addWidget(nan_group)
        main_layout.addLayout(btn_layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setLayout(main_layout)
        scroll.setWidget(content)
        
        tab.setLayout(QVBoxLayout())
        tab.layout().addWidget(scroll)
        self.tabs.addTab(tab, "Preprocess Data")
    
    def create_visualization_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout()
        
        # Top controls
        top_layout = QHBoxLayout()
        self.viz_dataset_combo = QComboBox()
        self.viz_dataset_combo.addItems(['Train', 'Test'])
        self.viz_dataset_combo.currentIndexChanged.connect(self.update_viz_columns)
        
        self.viz_combo = QComboBox()
        self.viz_combo.addItems(['Histogramma', 'Boxplot', 'Matrice di Correlazione'])
        self.viz_combo.currentIndexChanged.connect(self.update_visualizations)
        
        self.transform_combo = QComboBox()
        self.transform_combo.addItems(['Nessuna', 'Logaritmica', 'StandardScaler'])
        self.transform_btn = QPushButton('Applica Trasformazione')
        self.transform_btn.clicked.connect(self.apply_transformation)
        
        top_layout.addWidget(QLabel('Dataset:'))
        top_layout.addWidget(self.viz_dataset_combo)
        top_layout.addWidget(QLabel('Tipo Grafico:'))
        top_layout.addWidget(self.viz_combo)
        top_layout.addWidget(QLabel('Trasformazione:'))
        top_layout.addWidget(self.transform_combo)
        top_layout.addWidget(self.transform_btn)
        
        # Splitter for columns and plot
        splitter = QSplitter(Qt.Horizontal)
        
        # Column selection
        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QListWidget.MultiSelection)
        self.column_list.itemSelectionChanged.connect(self.update_visualizations)
        
        # Plot area
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        plot_scroll = QScrollArea()
        plot_scroll.setWidget(self.canvas)
        plot_scroll.setWidgetResizable(True)
        
        splitter.addWidget(self.column_list)
        splitter.addWidget(plot_scroll)
        splitter.setSizes([250, 750])
        
        main_layout.addLayout(top_layout)
        main_layout.addWidget(splitter)
        tab.setLayout(main_layout)
        self.tabs.addTab(tab, "Visualizzazioni")
    
    def create_model_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            'XGBoost Classificatore', 'XGBoost Regressore',
            'LightGBM Classificatore', 'LightGBM Regressore'
        ])
        
        # Target selection
        self.target_combo = QComboBox()
        
        # Training controls
        train_btn = QPushButton('Addestra Modello')
        train_btn.clicked.connect(self.train_model)
        
        # Results display
        self.results_text = QLabel()
        self.results_text.setWordWrap(True)
        
        layout.addWidget(QLabel('Seleziona Modello:'))
        layout.addWidget(self.model_combo)
        layout.addWidget(QLabel('Seleziona Variabile Target:'))
        layout.addWidget(self.target_combo)
        layout.addWidget(train_btn)
        layout.addWidget(self.results_text)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Modelli")
    
    def load_file(self, dataset_type):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, f"Carica Dataset {dataset_type}", "",
            "CSV Files (*.csv);;All Files (*)", options=options
        )
        
        if file_name:
            try:
                df = pd.read_csv(file_name)
                if dataset_type == 'train':
                    self.train_df = df
                else:
                    self.test_df = df
                
                self.update_preprocess_info()
                self.update_viz_columns()
                self.update_model_targets()
                QMessageBox.information(self, 'Successo', f'Dataset {dataset_type} caricato correttamente!')
            except Exception as e:
                QMessageBox.critical(self, 'Errore', f'Errore nel caricamento: {str(e)}')
    
    def update_preprocess_info(self):
        dataset = self.dataset_combo.currentText().lower()
        df = self.train_df if dataset == 'train' else self.test_df
        
        if df is None:
            self.info_text.setText('Nessun dataset caricato')
            self.nan_text.setText('')
            return
        
        # Update info
        buffer = io.StringIO()
        df.info(buf=buffer)
        self.info_text.setText(buffer.getvalue())
        
        # Update NaN counts
        self.nan_text.setText(
            'Valori mancanti per colonna:\n' + 
            '\n'.join([f'{col}: {count}' for col, count in df.isna().sum().items()])
        )
    
    def drop_nan(self):
        dataset = self.dataset_combo.currentText().lower()
        df = self.train_df if dataset == 'train' else self.test_df
        
        if df is not None:
            if dataset == 'train':
                self.train_df = df.dropna()
            else:
                self.test_df = df.dropna()
            self.update_preprocess_info()
    
    def impute_nan(self):
        dataset = self.dataset_combo.currentText().lower()
        df = self.train_df if dataset == 'train' else self.test_df
        
        if df is None:
            return
        
        method = self.impute_combo.currentText().lower()
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                if method == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif method == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Categorical columns
        cat_cols = df.select_dtypes(exclude=np.number).columns
        for col in cat_cols:
            if df[col].isna().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        self.update_preprocess_info()
    
    def update_viz_columns(self):
        dataset = self.viz_dataset_combo.currentText().lower()
        df = self.train_df if dataset == 'train' else self.test_df
        
        self.column_list.clear()
        if df is not None:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            self.column_list.addItems(numeric_cols)
    
    def update_visualizations(self):
        self.figure.clear()
        dataset = self.viz_dataset_combo.currentText().lower()
        df = self.train_df if dataset == 'train' else self.test_df
        
        if df is None or df.empty:
            return
        
        selected_cols = [item.text() for item in self.column_list.selectedItems()]
        if not selected_cols:
            return
        
        try:
            viz_type = self.viz_combo.currentText()
            df = df[selected_cols].select_dtypes(include=np.number)
            
            if viz_type == 'Histogramma':
                n_cols = 3
                n_rows = int(np.ceil(len(selected_cols) / n_cols))
                self.figure.subplots(n_rows, n_cols, figsize=(12, 8))
                
                for i, col in enumerate(selected_cols):
                    ax = self.figure.axes[i]
                    df[col].hist(ax=ax)
                    ax.set_title(col, fontsize=8)
                    ax.tick_params(axis='both', labelsize=6)
                
                for j in range(i+1, n_rows*n_cols):
                    self.figure.delaxes(self.figure.axes[j])
                
                self.figure.tight_layout()
            
            elif viz_type == 'Boxplot':
                self.figure.subplots(len(selected_cols), 1, figsize=(10, 2*len(selected_cols)))
                
                for i, col in enumerate(selected_cols):
                    ax = self.figure.axes[i]
                    df.boxplot(column=col, ax=ax)
                    ax.tick_params(axis='both', labelsize=6)
                
                self.figure.tight_layout()
            
            elif viz_type == 'Matrice di Correlazione':
                ax = self.figure.add_subplot(111)
                corr_matrix = df.corr()
                sns.heatmap(
                    corr_matrix, 
                    annot=True, 
                    fmt=".2f", 
                    cmap='coolwarm', 
                    mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
                    ax=ax
                )
                ax.set_title('Matrice di Correlazione', fontsize=12)
            
            self.canvas.draw()
        
        except Exception as e:
            print(f"Errore nella generazione del grafico: {str(e)}")
    
    def apply_transformation(self):
        dataset = self.viz_dataset_combo.currentText().lower()
        df = self.train_df if dataset == 'train' else self.test_df
        transform_type = self.transform_combo.currentText()
        
        if df is None:
            return
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        selected_cols = [item.text() for item in self.column_list.selectedItems()]
        
        try:
            if transform_type == 'Logaritmica':
                for col in selected_cols:
                    if (df[col] > 0).all():
                        df[col] = np.log(df[col])
            
            elif transform_type == 'StandardScaler':
                scaler = StandardScaler()
                df[selected_cols] = scaler.fit_transform(df[selected_cols])
            
            self.update_visualizations()
        
        except Exception as e:
            QMessageBox.warning(self, 'Errore', f'Trasformazione fallita: {str(e)}')
    
    def update_model_targets(self):
        self.target_combo.clear()
        if self.train_df is not None:
            self.target_combo.addItems(self.train_df.columns.tolist())
    
    def train_model(self):
        if self.train_df is None:
            QMessageBox.warning(self, 'Errore', 'Caricare prima il dataset di training!')
            return
        
        model_type = self.model_combo.currentText()
        target_col = self.target_combo.currentText()
        
        try:
            X = self.train_df.drop(target_col, axis=1)
            y = self.train_df[target_col]
            
            if 'Classificatore' in model_type:
                if 'XGBoost' in model_type:
                    model = XGBClassifier()
                else:
                    model = LGBMClassifier()
            else:
                if 'XGBoost' in model_type:
                    model = XGBRegressor()
                else:
                    model = LGBMRegressor()
            
            model.fit(X, y)
            score = model.score(X, y)
            self.results_text.setText(
                f'Modello addestrato con successo!\n'
                f'Tipo modello: {model_type}\n'
                f'Accuracy/R²: {score:.4f}'
            )
        
        except Exception as e:
            QMessageBox.critical(self, 'Errore', f'Addestramento fallito: {str(e)}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DataFrameViewer()
    sys.exit(app.exec_())