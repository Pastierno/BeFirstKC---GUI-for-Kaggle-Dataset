import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QComboBox, QScrollArea, QGroupBox, QLabel, QMessageBox,
    QListWidget, QSplitter, QLineEdit, QCheckBox, QPlainTextEdit
)
from PyQt5.QtCore import Qt
from sklearn.preprocessing import (
    StandardScaler, PowerTransformer, LabelEncoder, OneHotEncoder
)
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import seaborn as sns
import traceback
import io


class DataAnalysisTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.train_df = None
        self.test_df = None
        self.current_figsize = (10, 8)
        self.target_encoder = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Data Analysis Tool')
        self.setGeometry(100, 100, 1300, 900)
        self.tabs = QTabWidget()
        self.create_load_tab()
        self.create_preprocess_tab()
        self.create_encode_tab()
        self.create_visualize_tab()
        self.create_model_tab()
        self.setCentralWidget(self.tabs)
        self.show()

    # --- TAB 1: Load Data ---
    def create_load_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.btn_load_train = QPushButton('Load Train Dataset')
        self.btn_load_test = QPushButton('Load Test Dataset')
        self.btn_load_train.clicked.connect(lambda: self.load_file('train'))
        self.btn_load_test.clicked.connect(lambda: self.load_file('test'))
        btn_layout.addWidget(self.btn_load_train)
        btn_layout.addWidget(self.btn_load_test)

        self.text_head = QPlainTextEdit()
        self.text_head.setReadOnly(True)
        self.text_head.setPlaceholderText('DataFrame head will appear here...')

        layout.addLayout(btn_layout)
        layout.addWidget(self.text_head)
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Load Data')

    def load_file(self, which):
        fname, _ = QFileDialog.getOpenFileName(
            self, f'Load {which.title()} Dataset', '', 'CSV Files (*.csv);;All Files (*)'
        )
        if not fname:
            return
        try:
            df = pd.read_csv(fname)
            if which == 'train':
                self.train_df = df.copy()
            else:
                self.test_df = df.copy()
            self.text_head.setPlainText(df.head().to_string())
            self.update_all_lists()
            QMessageBox.information(self, 'Success', f'{which.title()} dataset loaded successfully')
        except Exception as e:
            QMessageBox.critical(self, 'Load Error', str(e))
            traceback.print_exc()

    # --- TAB 2: Preprocess ---
    def create_preprocess_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Dataset selector
        ds_layout = QHBoxLayout()
        ds_layout.addWidget(QLabel('Dataset:'))
        self.combo_pp_ds = QComboBox()
        self.combo_pp_ds.addItems(['Train', 'Test'])
        self.combo_pp_ds.currentIndexChanged.connect(self.update_pp_info)
        ds_layout.addWidget(self.combo_pp_ds)

        # Columns list and transforms
        splitter = QSplitter(Qt.Horizontal)
        self.list_pp_cols = QListWidget()
        self.list_pp_cols.setSelectionMode(QListWidget.MultiSelection)
        tr_group = QGroupBox('Transforms')
        tr_layout = QVBoxLayout()
        btn_yeo = QPushButton('Yeo-Johnson')
        btn_std = QPushButton('Standard Scaling')
        btn_yeo.clicked.connect(self.apply_yeo)
        btn_std.clicked.connect(self.apply_std)
        tr_layout.addWidget(btn_yeo)
        tr_layout.addWidget(btn_std)
        tr_group.setLayout(tr_layout)
        splitter.addWidget(self.list_pp_cols)
        splitter.addWidget(tr_group)

        # Info and NaN
        info_group = QGroupBox('Info & Missing')
        i_layout = QVBoxLayout()
        self.text_info = QPlainTextEdit(); self.text_info.setReadOnly(True)
        self.text_nan = QPlainTextEdit(); self.text_nan.setReadOnly(True)
        i_layout.addWidget(QLabel('Info:')); i_layout.addWidget(self.text_info)
        i_layout.addWidget(QLabel('Missing:')); i_layout.addWidget(self.text_nan)
        info_group.setLayout(i_layout)

        # Buttons
        btns = QHBoxLayout()
        btn_drop = QPushButton('Drop NaN'); btn_drop.clicked.connect(self.pp_dropna)
        btn_imp = QPushButton('Impute'); self.combo_imp = QComboBox(); self.combo_imp.addItems(['Mean','Median','Mode'])
        btn_imp.clicked.connect(self.pp_impute)
        btns.addWidget(btn_drop); btns.addWidget(self.combo_imp); btns.addWidget(btn_imp)

        layout.addLayout(ds_layout)
        layout.addWidget(splitter)
        layout.addWidget(info_group)
        layout.addLayout(btns)
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Preprocess')

    def update_pp_info(self):
        df = self.train_df if self.combo_pp_ds.currentText()=='Train' else self.test_df
        if df is None:
            self.list_pp_cols.clear(); self.text_info.clear(); self.text_nan.clear(); return
        # update columns
        self.list_pp_cols.clear(); self.list_pp_cols.addItems(df.columns)
        # info
        buf = io.StringIO(); df.info(buf=buf)
        self.text_info.setPlainText(buf.getvalue())
        # nan counts
        nan_text = '\n'.join([f'{c}: {n}' for c,n in df.isna().sum().items()])
        self.text_nan.setPlainText(nan_text)

    def apply_yeo(self):
        df = self.get_pp_df(); cols = [i.text() for i in self.list_pp_cols.selectedItems()]
        if df is None or not cols: return
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        try:
            df[cols] = pt.fit_transform(df[cols])
        except Exception as e:
            QMessageBox.warning(self, 'Transform Error', str(e))
        self.update_pp_info()

    def apply_std(self):
        df = self.get_pp_df(); cols = [i.text() for i in self.list_pp_cols.selectedItems()]
        if df is None or not cols: return
        sc = StandardScaler()
        try:
            df[cols] = sc.fit_transform(df[cols])
        except Exception as e:
            QMessageBox.warning(self, 'Transform Error', str(e))
        self.update_pp_info()

    def pp_dropna(self):
        df = self.get_pp_df();
        if df is not None:
            df.dropna(inplace=True)
            self.update_pp_info()

    def pp_impute(self):
        df = self.get_pp_df(); method = self.combo_imp.currentText()
        if df is None: return
        num = df.select_dtypes(include=np.number).columns
        for c in num:
            if method=='Mean': df[c].fillna(df[c].mean(), inplace=True)
            elif method=='Median': df[c].fillna(df[c].median(), inplace=True)
            else: df[c].fillna(df[c].mode()[0], inplace=True)
        self.update_pp_info()

    def get_pp_df(self):
        return self.train_df if self.combo_pp_ds.currentText()=='Train' else self.test_df

    # --- TAB 3: Encode Categorical ---
    def create_encode_tab(self):
        tab = QWidget(); layout = QVBoxLayout()
        layout.addWidget(QLabel('Dataset:'))
        self.combo_enc_ds = QComboBox(); self.combo_enc_ds.addItems(['Train','Test'])
        self.combo_enc_ds.currentIndexChanged.connect(self.update_enc)
        layout.addWidget(self.combo_enc_ds)
        self.list_enc = QListWidget(); self.list_enc.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(QLabel('Categorical Columns:')); layout.addWidget(self.list_enc)
        self.cb_label = QCheckBox('Label Encoding'); self.cb_onehot = QCheckBox('OneHot Encoding')
        layout.addWidget(self.cb_label); layout.addWidget(self.cb_onehot)
        btn = QPushButton('Apply'); btn.clicked.connect(self.apply_enc)
        layout.addWidget(btn); layout.addStretch()
        tab.setLayout(layout); self.tabs.addTab(tab,'Encode')

    def update_enc(self):
        df = self.train_df if self.combo_enc_ds.currentText()=='Train' else self.test_df
        self.list_enc.clear()
        if df is not None:
            cats = df.select_dtypes(exclude=np.number).columns
            self.list_enc.addItems(cats)

    def apply_enc(self):
        df = self.train_df if self.combo_enc_ds.currentText()=='Train' else self.test_df
        cols = [i.text() for i in self.list_enc.selectedItems()]
        if df is None or not cols:
            QMessageBox.warning(self,'Encode','No columns selected'); return
        if self.cb_label.isChecked():
            le=LabelEncoder()
            for c in cols:
                df[c]=le.fit_transform(df[c].astype(str))
        if self.cb_onehot.isChecked():
            ohe=OneHotEncoder(sparse=False, drop='first')
            arr=ohe.fit_transform(df[cols].astype(str))
            new_cols=ohe.get_feature_names_out(cols)
            df.drop(columns=cols, inplace=True)
            df[new_cols]=arr
        QMessageBox.information(self,'Encode','Encoding applied')
        self.update_enc(); self.update_pp_info(); self.update_all_lists()

    # --- TAB 4: Visualize ---
    def create_visualize_tab(self):
        tab=QWidget(); layout=QVBoxLayout()
        controls=QHBoxLayout()
        controls.addWidget(QLabel('Dataset:'))
        self.combo_viz_ds=QComboBox(); self.combo_viz_ds.addItems(['Train','Test']); self.combo_viz_ds.currentIndexChanged.connect(self.update_viz)
        controls.addWidget(self.combo_viz_ds)
        controls.addWidget(QLabel('Type:'))
        self.combo_viz_type=QComboBox(); self.combo_viz_type.addItems(['Histogram','Boxplot','Correlation']); controls.addWidget(self.combo_viz_type)
        controls.addStretch()
        layout.addLayout(controls)
        splitter=QSplitter(Qt.Horizontal)
        self.list_viz_cols=QListWidget(); self.list_viz_cols.setSelectionMode(QListWidget.MultiSelection)
        self.list_viz_cols.itemSelectionChanged.connect(self.plot_viz)
        splitter.addWidget(self.list_viz_cols)
        self.figure=Figure(figsize=self.current_figsize); self.canvas=FigureCanvas(self.figure)
        scroll=QScrollArea(); scroll.setWidget(self.canvas); scroll.setWidgetResizable(True)
        splitter.addWidget(scroll); splitter.setSizes([200,800])
        layout.addWidget(splitter)
        tab.setLayout(layout); self.tabs.addTab(tab,'Visualize')

    def update_viz(self):
        df = self.train_df if self.combo_viz_ds.currentText()=='Train' else self.test_df
        self.list_viz_cols.clear()
        if df is not None:
            self.list_viz_cols.addItems(df.select_dtypes(include=np.number).columns)

    def plot_viz(self):
        df=self.train_df if self.combo_viz_ds.currentText()=='Train' else self.test_df
        cols=[i.text() for i in self.list_viz_cols.selectedItems()]
        if df is None or not cols: return
        numdf=df[cols].select_dtypes(include=np.number)
        self.figure.clear()
        vt=self.combo_viz_type.currentText()
        ax=self.figure.add_subplot(111)
        try:
            if vt=='Histogram':
                numdf.hist(ax=ax)
            elif vt=='Boxplot':
                numdf.plot.box(ax=ax)
            else:
                corr=numdf.corr(); sns.heatmap(corr,mask=np.triu(np.ones_like(corr,dtype=bool)),ax=ax,annot=True)
        except Exception as e:
            QMessageBox.warning(self,'Plot Error',str(e))
        self.canvas.draw()

    # --- TAB 5: Model ---
    def create_model_tab(self):
        tab=QWidget(); layout=QVBoxLayout()
        layout.addWidget(QLabel('Drop Columns:'))
        self.list_model_drop=QListWidget(); self.list_model_drop.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.list_model_drop)
        btn_drop=QPushButton('Drop'); btn_drop.clicked.connect(self.model_drop)
        layout.addWidget(btn_drop)
        layout.addWidget(QLabel('Target:'))
        self.combo_target=QComboBox(); layout.addWidget(self.combo_target)
        layout.addWidget(QLabel('Model:'))
        self.combo_model=QComboBox(); self.combo_model.addItems(['XGBoost Classifier','XGBoost Regressor','LightGBM Classifier','LightGBM Regressor'])
        layout.addWidget(self.combo_model)
        btn_train=QPushButton('Train'); btn_train.clicked.connect(self.train_model)
        layout.addWidget(btn_train)
        self.text_model_res=QPlainTextEdit(); self.text_model_res.setReadOnly(True)
        layout.addWidget(self.text_model_res)
        tab.setLayout(layout); self.tabs.addTab(tab,'Model')

    def update_all_lists(self):
        # for model drop and target
        cols = list(self.train_df.columns) if self.train_df is not None else []
        self.list_model_drop.clear(); self.list_model_drop.addItems(cols)
        self.combo_target.clear(); self.combo_target.addItems(cols)
        # for preprocess
        self.update_pp_info(); # for encoding
        self.update_enc(); # for visualize
        self.update_viz()

    def model_drop(self):
        to_drop=[i.text() for i in self.list_model_drop.selectedItems()]
        if not to_drop:
            QMessageBox.warning(self,'Model','No columns selected'); return
        if self.train_df is not None: self.train_df.drop(columns=to_drop, inplace=True, errors='ignore')
        if self.test_df is not None: self.test_df.drop(columns=to_drop, inplace=True, errors='ignore')
        self.update_all_lists()
        QMessageBox.information(self,'Model',f'Dropped: {to_drop}')

    def train_model(self):
        if self.train_df is None:
            QMessageBox.warning(self,'Model','Load train data first'); return
        df=self.train_df.copy()
        target=self.combo_target.currentText()
        if target not in df.columns:
            QMessageBox.warning(self,'Model','Select valid target'); return
        X=df.drop(columns=[target])
        # drop non-numeric
        X=X.select_dtypes(include=np.number)
        y=df[target]
        is_clf='Classifier' in self.combo_model.currentText()
        if is_clf:
            self.target_encoder=LabelEncoder(); y=self.target_encoder.fit_transform(y.astype(str))
        # split
        X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)
        # instantiate
        model_name=self.combo_model.currentText()
        model=None
        if 'XGBoost' in model_name:
            model = XGBClassifier(eval_metric='mlogloss') if is_clf else XGBRegressor()
        else:
            model = LGBMClassifier() if is_clf else LGBMRegressor()
        # train
        try:
            model.fit(X_train,y_train)
            train_score=model.score(X_train,y_train)
            val_score=model.score(X_val,y_val)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            res = (
                f"Model: {model_name}\n"
                f"Train score: {train_score:.4f}\n"
                f"Validation score: {val_score:.4f}\n"
                f"CV mean: {cv_scores.mean():.4f} (std {cv_scores.std():.4f})"
            )
            self.text_model_res.setPlainText(res)
        except Exception as e:
            QMessageBox.critical(self,'Train Error',str(e))
            traceback.print_exc()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DataAnalysisTool()
    sys.exit(app.exec_())
