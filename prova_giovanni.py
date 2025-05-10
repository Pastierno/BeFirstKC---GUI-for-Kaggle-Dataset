import sys
import io
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox, QCheckBox, QLineEdit, QTextEdit)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

class DataAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Analysis App")
        self.train_df = None
        self.test_df = None
        self.transformed_df = None
        self.model = None
        self.scaler = None
        self.num_features = []
        self.feature_columns = None
        self.target_col = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout()

        # File selectors
        file_layout = QHBoxLayout()
        self.train_btn = QPushButton("Browse Train CSV")
        self.train_btn.clicked.connect(self.load_train)
        self.train_label = QLabel("Train: None")
        self.test_btn = QPushButton("Browse Test CSV")
        self.test_btn.clicked.connect(self.load_test)
        self.test_label = QLabel("Test: None")
        file_layout.addWidget(self.train_btn)
        file_layout.addWidget(self.train_label)
        file_layout.addWidget(self.test_btn)
        file_layout.addWidget(self.test_label)
        layout.addLayout(file_layout)

        # Target column selector
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target column:"))
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        target_layout.addWidget(self.target_combo)
        layout.addLayout(target_layout)

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
        viz_layout.addWidget(self.hist_check)
        viz_layout.addWidget(self.box_check)
        viz_layout.addWidget(self.corr_check)
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
        self.model_combo.addItems([
            "LogisticRegression", "RandomForestClassifier", "XGBClassifier", "LGBMClassifier",
            "LinearRegression", "RandomForestRegressor", "XGBRegressor", "LGBMRegressor"
        ])
        model_layout.addWidget(self.model_combo)
        self.train_model_btn = QPushButton("Train Model")
        self.train_model_btn.clicked.connect(self.train_model)
        model_layout.addWidget(self.train_model_btn)
        layout.addLayout(model_layout)

        # Predict
        pred_layout = QHBoxLayout()
        self.pred_btn = QPushButton("Predict Test & Save")
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
            self.target_combo.clear()
            self.target_combo.addItems(self.train_df.columns)
            self.target_combo.setEnabled(True)

    def load_test(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Test CSV", "", "CSV Files (*.csv)")
        if path:
            self.test_df = pd.read_csv(path)
            self.test_label.setText(f"Test: {path}")

    def preprocess(self):
        df = self.transformed_df
        method = self.null_combo.currentText()
        if method == "Drop Nulls":
            df = df.dropna()
        elif method.startswith("Fill"):
            if method == "Fill Mean": df = df.fillna(df.mean())
            if method == "Fill Median": df = df.fillna(df.median())
            if method == "Fill Mode": df = df.fillna(df.mode().iloc[0])
        if self.dup_check.isChecked():
            df = df.drop_duplicates()
        if self.std_check.isChecked():
            self.scaler = StandardScaler()
            numeric = df.select_dtypes(include=np.number)
            self.num_features = list(numeric.columns)
            df[self.num_features] = self.scaler.fit_transform(numeric)
        self.transformed_df = df

    def show_info(self):
        if self.transformed_df is not None:
            buf = io.StringIO()
            self.transformed_df.info(buf=buf)
            info_str = buf.getvalue()
            buf.close()
            self.output.setPlainText(info_str)

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
        num_df = self.transformed_df.select_dtypes(include=np.number)
        self.figure.clear()
        ax = self.figure.subplots()
        if self.hist_check.isChecked():
            num_df.plot(kind='hist', ax=ax)
        if self.box_check.isChecked():
            num_df.plot(kind='box', ax=ax)
        if self.corr_check.isChecked():
            import seaborn as sns
            sns.heatmap(num_df.corr(), annot=True, ax=ax)
        self.canvas.draw()

    def train_model(self):
        if self.transformed_df is None:
            self.output.append("Error: load training data first.")
            return
        self.target_col = self.target_combo.currentText()
        self.preprocess()
        X = self.transformed_df.drop(self.target_col, axis=1)
        y = self.transformed_df[self.target_col]
        X_processed = pd.get_dummies(X)
        self.feature_columns = X_processed.columns
        model_name = self.model_combo.currentText()
        models = {
            'LogisticRegression': LogisticRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'XGBClassifier': lambda: XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'LGBMClassifier': LGBMClassifier,
            'LinearRegression': LinearRegression,
            'RandomForestRegressor': RandomForestRegressor,
            'XGBRegressor': XGBRegressor,
            'LGBMRegressor': LGBMRegressor
        }
        self.model = models[model_name]()
        self.model.fit(X_processed, y)
        self.output.append(f"Trained {model_name} on target '{self.target_col}'")

    def predict(self):
        if self.model is None or self.test_df is None or self.feature_columns is None:
            return
        df_test = self.test_df.copy()
        # apply log / scale
        if self.log_input.text().strip():
            cols = [c.strip() for c in self.log_input.text().split(',')]
            for c in cols:
                df_test[c] = np.log1p(df_test[c])
        if self.std_check.isChecked() and self.num_features:
            for f in self.num_features:
                if f not in df_test:
                    df_test[f] = 0
            df_test[self.num_features] = self.scaler.transform(df_test[self.num_features])
        df_test_processed = pd.get_dummies(df_test)
        df_test_processed = df_test_processed.reindex(columns=self.feature_columns, fill_value=0)
        preds = self.model.predict(df_test_processed)
        # prepare submission
        df_test['prediction'] = preds
        # use id column and target_col
        if 'id' in df_test.columns and self.target_col:
            submission = pd.DataFrame({
                'id': df_test['id'],
                self.target_col: df_test['prediction']
            })
            path, _ = QFileDialog.getSaveFileName(self, "Save Submission CSV", "submission.csv", "CSV Files (*.csv)")
            if path:
                submission.to_csv(path, index=False)
                self.output.append(f"Submission saved to {path}")
        else:
            self.output.append("Error: 'id' column or target not found for submission.")
        self.output.append(str(df_test.head()))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = DataAnalysisApp()
    win.show()
    sys.exit(app.exec_())





