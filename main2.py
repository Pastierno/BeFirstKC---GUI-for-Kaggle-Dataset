import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QFileDialog, QComboBox, QScrollArea, 
                            QGroupBox, QLabel, QMessageBox, QListWidget, QSplitter, QLineEdit)
from PyQt5.QtCore import Qt
from sklearn.preprocessing import StandardScaler, PowerTransformer
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import seaborn as sns
import traceback
import io

class DataFrameViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.train_df = None
        self.test_df = None
        self.current_figsize = (10, 8)
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

    # TAB 1: File Loading
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

    # TAB 2: Preprocessing
    def create_preprocess_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout()
        
        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['Train', 'Test'])
        self.dataset_combo.currentIndexChanged.connect(self.update_preprocess_info)
        
        # Splitter for columns and transformations
        splitter = QSplitter(Qt.Horizontal)
        
        # Column list
        self.preprocess_col_list = QListWidget()
        self.preprocess_col_list.setSelectionMode(QListWidget.MultiSelection)
        
        # Transformations
        transform_group = QGroupBox("Data Transformations")
        transform_layout = QVBoxLayout()
        
        # Yeo-Johnson Transform
        self.log_transform_btn = QPushButton("Apply Yeo-Johnson Transform")
        self.log_transform_btn.clicked.connect(self.apply_power_transform)
        
        # Standard Scaler
        self.scaler_transform_btn = QPushButton("Apply Standard Scaling")
        self.scaler_transform_btn.clicked.connect(self.apply_scaler)
        
        transform_layout.addWidget(self.log_transform_btn)
        transform_layout.addWidget(self.scaler_transform_btn)
        transform_group.setLayout(transform_layout)
        
        splitter.addWidget(self.preprocess_col_list)
        splitter.addWidget(transform_group)
        
        # Info Section
        info_group = QGroupBox('Dataset Info')
        info_layout = QVBoxLayout()
        self.info_text = QLabel()
        self.info_text.setWordWrap(True)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        
        # NaN Section
        nan_group = QGroupBox('Missing Values')
        nan_layout = QVBoxLayout()
        self.nan_text = QLabel()
        self.nan_text.setWordWrap(True)
        nan_layout.addWidget(self.nan_text)
        nan_group.setLayout(nan_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.drop_nan_btn = QPushButton('Drop NaN')
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
        main_layout.addWidget(splitter)
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
        self.tabs.addTab(tab, "Preprocess")

    # TAB 3: Visualization
    def create_visualization_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout()
        
        # Controls
        top_layout = QHBoxLayout()
        self.viz_dataset_combo = QComboBox()
        self.viz_dataset_combo.addItems(['Train', 'Test'])
        self.viz_dataset_combo.currentIndexChanged.connect(self.update_viz_columns)
        
        self.viz_combo = QComboBox()
        self.viz_combo.addItems(['Histogram', 'Boxplot', 'Correlation Matrix'])
        self.viz_combo.currentIndexChanged.connect(self.update_visualizations)
        
        # Figure Size Controls
        size_layout = QHBoxLayout()
        self.width_input = QLineEdit('10')
        self.height_input = QLineEdit('8')
        self.apply_size_btn = QPushButton('Apply Size')
        self.apply_size_btn.clicked.connect(self.update_figsize)
        
        size_layout.addWidget(QLabel('Width:'))
        size_layout.addWidget(self.width_input)
        size_layout.addWidget(QLabel('Height:'))
        size_layout.addWidget(self.height_input)
        size_layout.addWidget(self.apply_size_btn)
        
        top_layout.addWidget(QLabel('Dataset:'))
        top_layout.addWidget(self.viz_dataset_combo)
        top_layout.addWidget(QLabel('Plot Type:'))
        top_layout.addWidget(self.viz_combo)
        top_layout.addLayout(size_layout)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Column List
        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QListWidget.MultiSelection)
        self.column_list.itemSelectionChanged.connect(self.update_visualizations)
        
        # Plot Area
        self.figure = Figure(figsize=self.current_figsize)
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
        self.tabs.addTab(tab, "Visualize")

    # TAB 4: Modeling
    def create_model_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            'XGBoost Classifier', 'XGBoost Regressor',
            'LightGBM Classifier', 'LightGBM Regressor'
        ])
        
        self.target_combo = QComboBox()
        
        train_btn = QPushButton('Train Model')
        train_btn.clicked.connect(self.train_model)
        
        self.results_text = QLabel()
        self.results_text.setWordWrap(True)
        
        layout.addWidget(QLabel('Model:'))
        layout.addWidget(self.model_combo)
        layout.addWidget(QLabel('Target:'))
        layout.addWidget(self.target_combo)
        layout.addWidget(train_btn)
        layout.addWidget(self.results_text)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Model")

    # Core functionality
    def load_file(self, dataset_type):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, f"Load {dataset_type} Dataset", "",
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
                self.update_preprocess_columns()
                QMessageBox.information(self, 'Success', f'{dataset_type} dataset loaded!')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Load error: {str(e)}')

    def update_preprocess_info(self):
        dataset = self.dataset_combo.currentText().lower()
        df = self.train_df if dataset == 'train' else self.test_df
        
        if df is None:
            self.info_text.setText('No dataset loaded')
            self.nan_text.setText('')
            return
        
        buffer = io.StringIO()
        df.info(buf=buffer)
        self.info_text.setText(buffer.getvalue())
        self.nan_text.setText('\n'.join([f'{col}: {count}' for col, count in df.isna().sum().items()]))

    def update_preprocess_columns(self):
        dataset = self.dataset_combo.currentText().lower()
        df = self.train_df if dataset == 'train' else self.test_df
        self.preprocess_col_list.clear()
        if df is not None:
            self.preprocess_col_list.addItems(df.columns.tolist())

    def apply_power_transform(self):
        dataset = self.dataset_combo.currentText().lower()
        df = self.train_df if dataset == 'train' else self.test_df
        selected_cols = [item.text() for item in self.preprocess_col_list.selectedItems()]
        transformer = PowerTransformer(method='yeo-johnson')
        for col in selected_cols:
            try:
                arr = df[[col]].values
                transformed = transformer.fit_transform(arr)
                df[col] = transformed.flatten()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Couldn't apply Yeo-Johnson to {col}: {str(e)}")
        self.update_preprocess_info()

    def apply_scaler(self):
        dataset = self.dataset_combo.currentText().lower()
        df = self.train_df if dataset == 'train' else self.test_df
        selected_cols = [item.text() for item in self.preprocess_col_list.selectedItems()]
        scaler = StandardScaler()
        for col in selected_cols:
            try:
                df[col] = scaler.fit_transform(df[[col]])
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Couldn't scale {col}: {str(e)}")
        self.update_preprocess_info()

    def drop_nan(self):
        dataset = self.dataset_combo.currentText().lower()
        if dataset == 'train' and self.train_df is not None:
            self.train_df = self.train_df.dropna()
        elif self.test_df is not None:
            self.test_df = self.test_df.dropna()
        self.update_preprocess_info()

    def impute_nan(self):
        dataset = self.dataset_combo.currentText().lower()
        df = self.train_df if dataset == 'train' else self.test_df
        
        if df is None:
            return
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns
        
        method = self.impute_combo.currentText().lower()
        for col in numeric_cols:
            if method == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif method == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        self.update_preprocess_info()

    def update_viz_columns(self):
        dataset = self.viz_dataset_combo.currentText().lower()
        df = self.train_df if dataset == 'train' else self.test_df
        self.column_list.clear()
        if df is not None:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            self.column_list.addItems(numeric_cols)

    def update_figsize(self):
        try:
            width = float(self.width_input.text())
            height = float(self.height_input.text())
            if width <= 0 or height <= 0:
                raise ValueError
            self.current_figsize = (width, height)
            self.figure.set_size_inches(width, height)
            self.canvas.draw()
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Invalid size values')

    def update_visualizations(self):
        try:
            self.figure.clear()
            dataset = self.viz_dataset_combo.currentText().lower()
            df = self.train_df if dataset == 'train' else self.test_df
            
            if df is None or df.empty:
                return
            
            selected_cols = [item.text() for item in self.column_list.selectedItems()]
            if not selected_cols:
                return
            
            df_num = df[selected_cols].select_dtypes(include=np.number)
            if df_num.empty:
                QMessageBox.warning(self, 'Warning', 'No numeric columns selected')
                return
            
            viz_type = self.viz_combo.currentText()
            
            if viz_type == 'Histogram':
                n_cols = 3
                n_rows = int(np.ceil(len(selected_cols) / n_cols))
                self.figure.clf()
                axes = self.figure.subplots(n_rows, n_cols)
                axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
                for i, col in enumerate(selected_cols):
                    ax = axes_flat[i]
                    sns.histplot(data=df_num, x=col, ax=ax)
                    ax.set_title(col, fontsize=8)
                    ax.tick_params(axis='both', labelsize=6)
                for j in range(len(selected_cols), len(axes_flat)):
                    axes_flat[j].axis('off')
                self.figure.tight_layout()

            elif viz_type == 'Boxplot':
                n_plots = len(selected_cols)
                self.figure.clf()
                axes = self.figure.subplots(n_plots, 1)
                axes = [axes] if n_plots == 1 else axes
                for ax, col in zip(axes, selected_cols):
                    df_num.boxplot(column=col, ax=ax, vert=False)
                    ax.tick_params(axis='both', labelsize=6)
                self.figure.tight_layout()

            elif viz_type == 'Correlation Matrix':
                self.figure.clf()
                ax = self.figure.add_subplot(111)
                corr_matrix = df_num.corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'shrink': 0.8}, ax=ax)
                ax.set_title('Correlation Matrix', fontsize=12)
                ax.tick_params(axis='x', rotation=45)

            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Plot error: {str(e)}')
            print(traceback.format_exc())

    def update_model_targets(self):
        self.target_combo.clear()
        if self.train_df is not None:
            self.target_combo.addItems(self.train_df.columns.tolist())

    def train_model(self):
        if self.train_df is None:
            QMessageBox.warning(self, 'Error', 'Load training data first')
            return
        
        try:
            model_type = self.model_combo.currentText()
            target_col = self.target_combo.currentText()
            
            X = self.train_df.drop(target_col, axis=1)
            y = self.train_df[target_col]
            
            if 'Classifier' in model_type:
                model = XGBClassifier() if 'XGBoost' in model_type else LGBMClassifier()
            else:
                model = XGBRegressor() if 'XGBoost' in model_type else LGBMRegressor()
            
            model.fit(X, y)
            score = model.score(X, y)
            self.results_text.setText(
                f'Trained {model_type}\n'
                f'Score: {score:.4f}\n'
                f'Target: {target_col}'
            )
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Training failed: {str(e)}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DataFrameViewer()
    sys.exit(app.exec_())
