import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox, QCheckBox, QLineEdit, QTextEdit)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class DataAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Analysis App")
        self.train_df = None
        self.test_df = None
        self.transformed_df = None
        self.model = None
        self.scaler = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout()

        # File selectors
        file_layout = QHBoxLayout()
        self.train_label = QLabel("Train: None")
        self.train_btn = QPushButton("Browse Train CSV")
        self.train_btn.clicked.connect(self.load_train)
        self.test_label = QLabel("Test: None")
        self.test_btn = QPushButton("Browse Test CSV")
        self.test_btn.clicked.connect(self.load_test)
        file_layout.addWidget(self.train_btn)
        file_layout.addWidget(self.train_label)
        file_layout.addWidget(self.test_btn)
        file_layout.addWidget(self.test_label)
        layout.addLayout(file_layout)

        # Null handling
        null_layout = QHBoxLayout()
        null_layout.addWidget(QLabel("Nulls:"))
        self.null_combo = QComboBox()
        self.null_combo.addItems(["None","Drop Nulls","Fill Mean","Fill Median","Fill Mode"])
        null_layout.addWidget(self.null_combo)
        layout.addLayout(null_layout)

        # Duplicates
        dup_layout = QHBoxLayout()
        self.dup_check = QCheckBox("Drop Duplicates")
        dup_layout.addWidget(self.dup_check)
        layout.addLayout(dup_layout)

        # Info display
        info_layout = QHBoxLayout()
        self.info_btn = QPushButton("Show DataFrame Info")
        self.info_btn.clicked.connect(self.show_info)
        info_layout.addWidget(self.info_btn)
        layout.addLayout(info_layout)

        # Dtype conversion
        dtype_layout = QHBoxLayout()
        dtype_layout.addWidget(QLabel("Convert dtype (col1,col2)->dtype:"))
        self.dtype_input = QLineEdit()
        dtype_layout.addWidget(self.dtype_input)
        self.dtype_btn = QPushButton("Apply Dtype")
        self.dtype_btn.clicked.connect(self.apply_dtype)
        dtype_layout.addWidget(self.dtype_btn)
        layout.addLayout(dtype_layout)

        # Log transform
        log_layout = QHBoxLayout()
        log_layout.addWidget(QLabel("Log transform cols (col1,col2):"))
        self.log_input = QLineEdit()
        log_layout.addWidget(self.log_input)
        self.log_btn = QPushButton("Apply Log Transform")
        self.log_btn.clicked.connect(self.apply_log)
        log_layout.addWidget(self.log_btn)
        layout.addLayout(log_layout)

        # Standardize
        std_layout = QHBoxLayout()
        self.std_check = QCheckBox("Standardize")
        std_layout.addWidget(self.std_check)
        layout.addLayout(std_layout)

        # Visualization options
        viz_layout = QHBoxLayout()
        self.hist_check = QCheckBox("Histplot")
        self.box_check = QCheckBox("Boxplot")
        self.corr_check = QCheckBox("Corr Matrix")
        self.scatter_check = QCheckBox("Scatter Matrix")
        viz_layout.addWidget(self.hist_check)
        viz_layout.addWidget(self.box_check)
        viz_layout.addWidget(self.corr_check)
        viz_layout.addWidget(self.scatter_check)
        self.viz_btn = QPushButton("Plot")
        self.viz_btn.clicked.connect(self.plot)
        viz_layout.addWidget(self.viz_btn)
        layout.addLayout(viz_layout)

        # Matplotlib canvas
        self.figure = Figure(figsize=(8,6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LogisticRegression","RandomForest","XGBoost","LightGBM"])
        model_layout.addWidget(self.model_combo)
        self.train_btn2 = QPushButton("Train Model")
        self.train_btn2.clicked.connect(self.train_model)
        model_layout.addWidget(self.train_btn2)
        layout.addLayout(model_layout)

        # Predict
        pred_layout = QHBoxLayout()
        self.pred_btn = QPushButton("Predict Test")
        self.pred_btn.clicked.connect(self.predict)
        pred_layout.addWidget(self.pred_btn)
        layout.addLayout(pred_layout)

        # Output
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

        central.setLayout(layout)
        self.setCentralWidget(central)

    def load_train(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Train CSV", "", "CSV Files (*.csv)")
        if path:
            self.train_df = pd.read_csv(path)
            self.transformed_df = self.train_df.copy()
            self.train_label.setText(f"Train: {path}")

    def load_test(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Test CSV", "", "CSV Files (*.csv)")
        if path:
            self.test_df = pd.read_csv(path)
            self.test_label.setText(f"Test: {path}")

    def preprocess(self):
        df = self.transformed_df
        # Handle nulls
        method = self.null_combo.currentText()
        if method == "Drop Nulls":
            df = df.dropna()
        elif method.startswith("Fill"):
            if method == "Fill Mean": df = df.fillna(df.mean())
            if method == "Fill Median": df = df.fillna(df.median())
            if method == "Fill Mode": df = df.fillna(df.mode().iloc[0])
        # Duplicates
        if self.dup_check.isChecked(): df = df.drop_duplicates()
        # Standardize
        if self.std_check.isChecked():
            self.scaler = StandardScaler()
            numeric = df.select_dtypes(include=np.number)
            df[numeric.columns] = self.scaler.fit_transform(numeric)
        self.transformed_df = df

    def show_info(self):
        if self.transformed_df is not None:
            buf = []
            self.transformed_df.info(buf=buf)
            self.output.setPlainText("".join(buf))

    def apply_dtype(self):
        text = self.dtype_input.text().strip()
        try:
            cols, dtype = text.split('->')
            cols = [c.strip() for c in cols.split(',')]
            self.transformed_df[cols] = self.transformed_df[cols].astype(dtype)
            self.output.append(f"Converted {cols} to {dtype}")
        except Exception as e:
            self.output.append(f"Error converting dtype: {e}")

    def apply_log(self):
        text = self.log_input.text().strip()
        cols = [c.strip() for c in text.split(',')]
        for c in cols:
            self.transformed_df[c] = np.log1p(self.transformed_df[c])
        self.output.append(f"Applied log transform to {cols}")

    def plot(self):
        self.preprocess()
        df = self.transformed_df
        self.figure.clear()
        ax = self.figure.subplots()
        # Hist
        if self.hist_check.isChecked():
            df.hist(ax=ax)
        # Boxplot
        if self.box_check.isChecked():
            df.plot.box(ax=ax)
        # Correlation
        if self.corr_check.isChecked():
            import seaborn as sns
            sns.heatmap(df.corr(), annot=True, ax=ax)
        # Scatter matrix
        if self.scatter_check.isChecked():
            scatter_matrix(df, ax=ax)
        self.canvas.draw()

    def train_model(self):
        if self.transformed_df is None:
            return
        self.preprocess()
        X = self.transformed_df.drop('target', axis=1, errors='ignore')
        y = self.transformed_df['target'] if 'target' in self.transformed_df else None
        model_name = self.model_combo.currentText()
        if model_name == 'LogisticRegression':
            self.model = LogisticRegression()
        elif model_name == 'RandomForest':
            self.model = RandomForestClassifier()
        elif model_name == 'XGBoost':
            self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        elif model_name == 'LightGBM':
            self.model = LGBMClassifier()
        self.model.fit(X, y)
        self.output.append(f"Trained {model_name}")

    def predict(self):
        if self.model is None or self.test_df is None:
            return
        df_test = self.test_df.copy()
        # Apply same transforms
        # Nulls, dtypes, log, standardization
        # For simplicity, apply preprocessing on train pipeline and then scaler to test
        if self.log_input.text().strip():
            cols = [c.strip() for c in self.log_input.text().split(',')]
            for c in cols:
                df_test[c] = np.log1p(df_test[c])
        if self.std_check.isChecked():
            numeric = df_test.select_dtypes(include=np.number)
            df_test[numeric.columns] = self.scaler.transform(numeric)
        preds = self.model.predict(df_test)
        df_test['prediction'] = preds
        self.output.append(str(df_test.head()))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = DataAnalysisApp()
    win.show()
    sys.exit(app.exec_())
