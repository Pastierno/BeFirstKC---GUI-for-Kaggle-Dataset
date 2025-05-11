import sys, io, pandas as pd, numpy as np, optuna, seaborn as sns, traceback, pickle
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QComboBox, QScrollArea,
    QGroupBox, QLabel, QMessageBox, QListWidget, QSplitter, QSpinBox,
    QProgressDialog, QPlainTextEdit, QCheckBox)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, confusion_matrix)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

class DataAnalysisTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.train_df = None
        self.test_df = None
        self.submission_df = None  # Copia integra del test per l'ID
        self.last_loaded = None
        self.target_encoder = None
        self.model = None
        self.best_params = None
        self.feature_cols = None
        self.current_figsize = (10, 8)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Data Analysis Tool')
        self.setGeometry(100, 100, 1300, 900)
        self.tabs = QTabWidget()
        self.create_load_tab()
        self.create_preprocess_tab()
        self.create_cat_impute_tab()
        self.create_encode_tab()
        self.create_visualize_tab()
        self.create_model_tab()
        self.create_submission_tab()
        self.setCentralWidget(self.tabs)
        self.show()

    # --- Load Data ---
    def create_load_tab(self):
        t = QWidget(); l = QVBoxLayout()
        h = QHBoxLayout()
        self.btn_load_train = QPushButton('Load Train Dataset')
        self.btn_load_test  = QPushButton('Load Test Dataset')
        self.btn_load_train.clicked.connect(lambda: self.load_file('train'))
        self.btn_load_test .clicked.connect(lambda: self.load_file('test'))
        h.addWidget(self.btn_load_train); h.addWidget(self.btn_load_test)
        l.addLayout(h)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel('Rows to show:'))
        self.spin_head_rows = QSpinBox(); self.spin_head_rows.setRange(1, 1000); self.spin_head_rows.setValue(5)
        self.spin_head_rows.valueChanged.connect(self.update_head_view)
        h2.addWidget(self.spin_head_rows); h2.addStretch()
        l.addLayout(h2)
        self.text_head = QPlainTextEdit(); self.text_head.setReadOnly(True)
        l.addWidget(self.text_head)
        t.setLayout(l)
        self.tabs.addTab(t, 'Load Data')

    def load_file(self, which):
        fn, _ = QFileDialog.getOpenFileName(self, f'Load {which.title()} Dataset', '', 'CSV Files (*.csv);;All Files (*)')
        if not fn: return
        try:
            df = pd.read_csv(fn)
            if which == 'train':
                self.train_df = df.copy()
            else:
                self.test_df       = df.copy()
                self.submission_df = df.copy()  # Salva copia intera per l'ID
            self.last_loaded = which
            self.update_all_lists()
            self.update_head_view()
            QMessageBox.information(self, 'Success', f'{which.title()} dataset loaded successfully')
        except Exception as e:
            QMessageBox.critical(self, 'Load Error', str(e))
            traceback.print_exc()

    def update_head_view(self):
        if not self.last_loaded: return
        df = (self.train_df if self.last_loaded=='train' else self.test_df)
        if df is None: return
        n = self.spin_head_rows.value()
        self.text_head.setPlainText(df.head(n).to_string())

    # --- Preprocess Numeric ---
    def create_preprocess_tab(self):
        t = QWidget(); l = QVBoxLayout()
        ds = QHBoxLayout(); ds.addWidget(QLabel('Dataset:'))
        self.combo_pp_ds = QComboBox(); self.combo_pp_ds.addItems(['Train','Test'])
        self.combo_pp_ds.currentIndexChanged.connect(self.update_pp_info)
        ds.addWidget(self.combo_pp_ds); l.addLayout(ds)

        sp = QSplitter(Qt.Horizontal)
        sw = QWidget(); sl = QVBoxLayout()
        sa = QPushButton('Select All'); da = QPushButton('Deselect All')
        sa.clicked.connect(lambda: self.list_pp_cols.selectAll())
        da.clicked.connect(lambda: self.list_pp_cols.clearSelection())
        sl.addWidget(sa); sl.addWidget(da)
        self.list_pp_cols = QListWidget(); self.list_pp_cols.setSelectionMode(QListWidget.MultiSelection)
        sl.addWidget(self.list_pp_cols); sw.setLayout(sl)

        tg = QGroupBox('Transforms'); tl = QVBoxLayout()
        yb = QPushButton('Yeo-Johnson'); ss = QPushButton('Standard Scaling')
        yb.clicked.connect(self.apply_yeo); ss.clicked.connect(self.apply_std)
        tl.addWidget(yb); tl.addWidget(ss); tg.setLayout(tl)

        sp.addWidget(sw); sp.addWidget(tg)
        l.addWidget(sp)

        ig = QGroupBox('Info & Missing'); il = QVBoxLayout()
        self.text_info = QPlainTextEdit(); self.text_info.setReadOnly(True)
        self.text_nan  = QPlainTextEdit(); self.text_nan.setReadOnly(True)
        il.addWidget(QLabel('Info:'));    il.addWidget(self.text_info)
        il.addWidget(QLabel('Missing:')); il.addWidget(self.text_nan)
        ig.setLayout(il); l.addWidget(ig)

        hb = QHBoxLayout()
        dn = QPushButton('Drop NaN'); dn.clicked.connect(self.pp_dropna)
        im = QComboBox(); im.addItems(['Mean','Median','Mode'])
        ib = QPushButton('Impute Numeric'); ib.clicked.connect(self.pp_impute)
        hb.addWidget(dn); hb.addWidget(im); hb.addWidget(ib)
        self.combo_imp = im; l.addLayout(hb)

        t.setLayout(l); self.tabs.addTab(t,'Preprocess')

    def update_pp_info(self):
        df = self.train_df if self.combo_pp_ds.currentText()=='Train' else self.test_df
        if df is None:
            self.list_pp_cols.clear(); self.text_info.clear(); self.text_nan.clear(); return
        self.list_pp_cols.clear(); self.list_pp_cols.addItems(df.columns)
        buf = io.StringIO(); df.info(buf=buf)
        self.text_info.setPlainText(buf.getvalue())
        self.text_nan.setPlainText('\n'.join(f'{c}: {n}' for c,n in df.isna().sum().items()))

    def apply_yeo(self):
        df = self.train_df if self.combo_pp_ds.currentText()=='Train' else self.test_df
        cols = [i.text() for i in self.list_pp_cols.selectedItems()]
        if df is None or not cols: return
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        try:
            df[cols] = pt.fit_transform(df[cols])
        except Exception as e:
            QMessageBox.warning(self, 'Transform Error', str(e))
        self.update_pp_info(); self.update_viz()

    def apply_std(self):
        df = self.train_df if self.combo_pp_ds.currentText()=='Train' else self.test_df
        cols = [i.text() for i in self.list_pp_cols.selectedItems()]
        if df is None or not cols: return
        sc = StandardScaler()
        try:
            df[cols] = sc.fit_transform(df[cols])
        except Exception as e:
            QMessageBox.warning(self, 'Transform Error', str(e))
        self.update_pp_info(); self.update_viz()

    def pp_dropna(self):
        df = self.train_df if self.combo_pp_ds.currentText()=='Train' else self.test_df
        if df is not None:
            df.dropna(inplace=True)
            self.update_pp_info()

    def pp_impute(self):
        df = self.train_df if self.combo_pp_ds.currentText()=='Train' else self.test_df
        m = self.combo_imp.currentText()
        if df is None: return
        for c in df.select_dtypes(include=np.number).columns:
            if m=='Mean':   df[c].fillna(df[c].mean(), inplace=True)
            elif m=='Median': df[c].fillna(df[c].median(), inplace=True)
            else:            df[c].fillna(df[c].mode()[0], inplace=True)
        self.update_pp_info()

    # --- Categorical Impute ---
    def create_cat_impute_tab(self):
        t = QWidget(); l = QVBoxLayout()
        ds = QHBoxLayout(); ds.addWidget(QLabel('Dataset:'))
        self.combo_cat_imp_ds = QComboBox(); self.combo_cat_imp_ds.addItems(['Train','Test'])
        self.combo_cat_imp_ds.currentIndexChanged.connect(self.update_cat_imp)
        ds.addWidget(self.combo_cat_imp_ds); l.addLayout(ds)

        hb = QHBoxLayout()
        sa = QPushButton('Select All'); da = QPushButton('Deselect All')
        sa.clicked.connect(lambda: self.list_cat_imp_cols.selectAll())
        da.clicked.connect(lambda: self.list_cat_imp_cols.clearSelection())
        hb.addWidget(sa); hb.addWidget(da); l.addLayout(hb)

        self.list_cat_imp_cols = QListWidget(); self.list_cat_imp_cols.setSelectionMode(QListWidget.MultiSelection)
        l.addWidget(QLabel('Categorical Columns:')); l.addWidget(self.list_cat_imp_cols)

        btn = QPushButton('Impute Mode'); btn.clicked.connect(self.apply_cat_impute)
        l.addWidget(btn)
        t.setLayout(l)
        self.tabs.addTab(t,'Impute Cat')

    def update_cat_imp(self):
        df = self.train_df if self.combo_cat_imp_ds.currentText()=='Train' else self.test_df
        self.list_cat_imp_cols.clear()
        if df is not None:
            self.list_cat_imp_cols.addItems(df.select_dtypes(exclude=np.number).columns)

    def apply_cat_impute(self):
        df = self.train_df if self.combo_cat_imp_ds.currentText()=='Train' else self.test_df
        cols = [i.text() for i in self.list_cat_imp_cols.selectedItems()]
        if df is None or not cols:
            QMessageBox.warning(self, 'Impute Cat', 'No columns selected'); return
        for c in cols:
            df[c].fillna(df[c].mode()[0], inplace=True)
        QMessageBox.information(self, 'Impute Cat', f'Mode imputed: {cols}')
        self.update_pp_info(); self.update_enc(); self.update_viz(); self.update_all_lists()

    # --- Encode ---
    def create_encode_tab(self):
        t = QWidget(); l = QVBoxLayout()
        l.addWidget(QLabel('Dataset:'))
        self.combo_enc_ds = QComboBox(); self.combo_enc_ds.addItems(['Train','Test'])
        self.combo_enc_ds.currentIndexChanged.connect(self.update_enc)
        l.addWidget(self.combo_enc_ds)

        hb = QHBoxLayout()
        sa = QPushButton('Select All'); da = QPushButton('Deselect All')
        sa.clicked.connect(lambda: self.list_enc.selectAll()); da.clicked.connect(lambda: self.list_enc.clearSelection())
        hb.addWidget(sa); hb.addWidget(da); l.addLayout(hb)

        self.list_enc = QListWidget(); self.list_enc.setSelectionMode(QListWidget.MultiSelection)
        l.addWidget(QLabel('Categorical Columns:')); l.addWidget(self.list_enc)

        self.cb_label  = QCheckBox('Label Encoding')
        self.cb_onehot = QCheckBox('OneHot Encoding')
        l.addWidget(self.cb_label); l.addWidget(self.cb_onehot)

        btn = QPushButton('Apply Encoding'); btn.clicked.connect(self.apply_enc)
        l.addWidget(btn)
        t.setLayout(l)
        self.tabs.addTab(t,'Encode')

    def update_enc(self):
        df = self.train_df if self.combo_enc_ds.currentText()=='Train' else self.test_df
        self.list_enc.clear()
        if df is not None:
            self.list_enc.addItems(df.select_dtypes(exclude=np.number).columns)

    def apply_enc(self):
        df = self.train_df if self.combo_enc_ds.currentText()=='Train' else self.test_df
        cols = [i.text() for i in self.list_enc.selectedItems()]
        if df is None or not cols:
            QMessageBox.warning(self, 'Encode', 'No columns selected'); return
        if self.cb_label.isChecked():
            le = LabelEncoder()
            for c in cols: df[c] = le.fit_transform(df[c].astype(str))
        if self.cb_onehot.isChecked():
            ohe = OneHotEncoder(sparse_output=False, drop='first')
            arr = ohe.fit_transform(df[cols].astype(str))
            new_cols = ohe.get_feature_names_out(cols)
            df_ohe = pd.DataFrame(arr, columns=new_cols, index=df.index)
            df.drop(columns=cols, inplace=True)
            df = pd.concat([df.reset_index(drop=True), df_ohe.reset_index(drop=True)], axis=1)
            if self.combo_enc_ds.currentText()=='Train':
                self.train_df = df
            else:
                self.test_df = df
        QMessageBox.information(self, 'Encode', 'Applied')
        self.update_enc(); self.update_pp_info(); self.update_all_lists()

    # --- Visualize ---
    def create_visualize_tab(self):
        t = QWidget(); l = QVBoxLayout()
        c = QHBoxLayout(); c.addWidget(QLabel('Dataset:'))
        self.combo_viz_ds = QComboBox(); self.combo_viz_ds.addItems(['Train','Test'])
        self.combo_viz_ds.currentIndexChanged.connect(self.update_viz); c.addWidget(self.combo_viz_ds)
        c.addWidget(QLabel('Type:'))
        self.combo_viz_type = QComboBox(); self.combo_viz_type.addItems(['Histogram','Boxplot','Correlation'])
        c.addWidget(self.combo_viz_type); c.addStretch(); l.addLayout(c)

        sp = QSplitter(Qt.Horizontal)
        sw = QWidget(); sl = QVBoxLayout()
        sa = QPushButton('Select All'); da = QPushButton('Deselect All')
        sa.clicked.connect(lambda: self.list_viz_cols.selectAll()); da.clicked.connect(lambda: self.list_viz_cols.clearSelection())
        sl.addWidget(sa); sl.addWidget(da)
        self.list_viz_cols = QListWidget(); self.list_viz_cols.setSelectionMode(QListWidget.MultiSelection)
        self.list_viz_cols.itemSelectionChanged.connect(self.plot_viz)
        sl.addWidget(self.list_viz_cols); sw.setLayout(sl); sp.addWidget(sw)

        self.figure = Figure(figsize=self.current_figsize); self.canvas = FigureCanvas(self.figure)
        scr = QScrollArea(); scr.setWidget(self.canvas); scr.setWidgetResizable(True); sp.addWidget(scr)
        sp.setSizes([200,800]); l.addWidget(sp); t.setLayout(l); self.tabs.addTab(t,'Visualize')

    def update_viz(self):
        df = self.train_df if self.combo_viz_ds.currentText()=='Train' else self.test_df
        self.list_viz_cols.clear()
        if df is not None:
            self.list_viz_cols.addItems(df.select_dtypes(include=np.number).columns)

    def plot_viz(self):
        df = self.train_df if self.combo_viz_ds.currentText()=='Train' else self.test_df
        cols = [i.text() for i in self.list_viz_cols.selectedItems()]
        if df is None or not cols: return
        numdf = df[cols].select_dtypes(include=np.number)
        self.figure.clear(); ax = self.figure.add_subplot(111)
        vt = self.combo_viz_type.currentText()
        try:
            if vt == 'Histogram':
                for c in cols: ax.hist(numdf[c].dropna(), bins=10, alpha=0.5, label=c)
                ax.legend()
            elif vt == 'Boxplot':
                ax.boxplot([numdf[c].dropna() for c in cols], labels=cols)
            else:
                corr = numdf.corr()
                sns.heatmap(corr, mask=np.triu(np.ones_like(corr,bool)), ax=ax, annot=False, cmap='viridis')
                ax.tick_params(axis='both', labelsize=max(6,12-len(cols)))
        except Exception as e:
            QMessageBox.warning(self, 'Plot Error', str(e))
        self.canvas.draw()

    # --- Model & Optuna ---
    def create_model_tab(self):
        t = QWidget(); l = QVBoxLayout()
        l.addWidget(QLabel('Drop Columns:'))
        hb = QHBoxLayout()
        sa = QPushButton('Select All'); da = QPushButton('Deselect All')
        sa.clicked.connect(lambda: self.list_model_drop.selectAll()); da.clicked.connect(lambda: self.list_model_drop.clearSelection())
        hb.addWidget(sa); hb.addWidget(da); l.addLayout(hb)
        self.list_model_drop = QListWidget(); self.list_model_drop.setSelectionMode(QListWidget.MultiSelection)
        l.addWidget(self.list_model_drop)

        db = QPushButton('Drop'); db.clicked.connect(self.model_drop); l.addWidget(db)
        l.addWidget(QLabel('Target:')); self.combo_target = QComboBox(); l.addWidget(self.combo_target)
        l.addWidget(QLabel('Model:'));  self.combo_model  = QComboBox()
        self.combo_model.addItems(['XGBoost Classifier','XGBoost Regressor','LightGBM Classifier','LightGBM Regressor'])
        l.addWidget(self.combo_model)

        tb = QPushButton('Train'); tb.clicked.connect(self.train_model); l.addWidget(tb)
        hb2 = QHBoxLayout(); hb2.addWidget(QLabel('Trials:'))
        self.spin_trials = QSpinBox(); self.spin_trials.setRange(1,500); self.spin_trials.setValue(50)
        hb2.addWidget(self.spin_trials)
        op = QPushButton('Optimize'); op.clicked.connect(self.optimize_model); hb2.addWidget(op)
        l.addLayout(hb2)

        self.btn_train_best = QPushButton('Train Best Params'); self.btn_train_best.setEnabled(False)
        self.btn_train_best.clicked.connect(self.train_best); l.addWidget(self.btn_train_best)
        self.btn_save_model  = QPushButton('Save Model');        self.btn_save_model.setEnabled(False)
        self.btn_save_model.clicked.connect(self.save_model);    l.addWidget(self.btn_save_model)

        self.text_model_res = QPlainTextEdit(); self.text_model_res.setReadOnly(True); l.addWidget(self.text_model_res)
        t.setLayout(l); self.tabs.addTab(t,'Model')

    def update_all_lists(self):
        cols = list(self.train_df.columns) if self.train_df is not None else []
        for w in [self.list_pp_cols, self.list_model_drop]:
            w.clear(); w.addItems(cols)
        self.combo_target.clear(); self.combo_target.addItems(cols)
        ids = list(self.submission_df.columns) if self.submission_df is not None else cols
        self.combo_id.clear(); self.combo_id.addItems(ids)
        self.update_pp_info(); self.update_enc(); self.update_viz()

    def model_drop(self):
        to_drop = [i.text() for i in self.list_model_drop.selectedItems()]
        if not to_drop:
            QMessageBox.warning(self,'Model','No columns selected'); return
        if self.train_df is not None:
            self.train_df.drop(columns=to_drop, inplace=True, errors='ignore')
        if self.test_df is not None:
            self.test_df.drop(columns=to_drop, inplace=True, errors='ignore')
        # submission_df mantiene intatta la colonna ID
        self.update_all_lists()
        QMessageBox.information(self,'Model',f'Dropped {to_drop}')

    def train_model(self):
        if self.train_df is None:
            QMessageBox.warning(self,'Model','Load train data first'); return
        df  = self.train_df.copy(); tgt = self.combo_target.currentText()
        if tgt not in df.columns:
            QMessageBox.warning(self,'Model','Select target'); return
        X = df.drop(columns=[tgt]).select_dtypes(include=np.number); y = df[tgt]
        is_clf = 'Classifier' in self.combo_model.currentText()
        if is_clf:
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y.astype(str))
        self.feature_cols = X.columns.tolist()
        Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
        mname = self.combo_model.currentText()
        if 'XGBoost' in mname:
            self.model = XGBClassifier(eval_metric='mlogloss') if is_clf else XGBRegressor()
        else:
            self.model = LGBMClassifier() if is_clf else LGBMRegressor()
        dlg = QProgressDialog('Training...', None, 0, 0, self)
        dlg.setWindowModality(Qt.WindowModal); dlg.show(); QApplication.processEvents()
        try:
            self.model.fit(Xtr, ytr); dlg.reset()
            yt = self.model.predict(Xtr); yv_pred = self.model.predict(Xv)
            s = f'Model: {mname}\n'
            if is_clf:
                s += (
                    f"Train acc: {accuracy_score(ytr, yt):.4f}\n"
                    f"Val acc: {accuracy_score(yv, yv_pred):.4f}\n"
                    f"Prec(w): {precision_score(yv, yv_pred, average='weighted'):.4f}\n"
                    f"Rec(w): {recall_score(yv, yv_pred, average='weighted'):.4f}\n"
                    f"CM:\n{confusion_matrix(yv, yv_pred)}\n"
                )
            else:
                r2t = self.model.score(Xtr, ytr); r2v = self.model.score(Xv, yv_pred)
                mae = mean_absolute_error(yv, yv_pred); mse = mean_squared_error(yv, yv_pred)
                rmse = np.sqrt(mse)
                s += (
                    f"Train R2: {r2t:.4f}\n"
                    f"Val R2: {r2v:.4f}\n"
                    f"MAE: {mae:.4f}\n"
                    f"MSE: {mse:.4f}\n"
                    f"RMSE: {rmse:.4f}\n"
                )
            cv = cross_val_score(self.model, Xtr, ytr, cv=5)
            s += f"CV mean: {cv.mean():.4f} (std {cv.std():.4f})"
            self.text_model_res.setPlainText(s)
            self.btn_save_model.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self,'Train Error',str(e)); traceback.print_exc()

    def optimize_model(self):
        if self.train_df is None:
            QMessageBox.warning(self,'Model','Load train data first'); return
        df  = self.train_df.copy(); tgt = self.combo_target.currentText()
        if tgt not in df.columns:
            QMessageBox.warning(self,'Model','Select target'); return
        X = df.drop(columns=[tgt]).select_dtypes(include=np.number); y = df[tgt]
        is_clf = 'Classifier' in self.combo_model.currentText()
        if is_clf and self.target_encoder:
            y = self.target_encoder.transform(y.astype(str))
        self.feature_cols = X.columns.tolist()
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        trials = self.spin_trials.value();  mname = self.combo_model.currentText()
        progress = QProgressDialog('Optuna...', 'Cancel', 0, trials, self)
        progress.setWindowModality(Qt.WindowModal); progress.show()
        def obj(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }
            if 'XGBoost' in mname:
                model = XGBClassifier(**params, eval_metric='mlogloss') if is_clf else XGBRegressor(**params)
            else:
                model = LGBMClassifier(**params) if is_clf else LGBMRegressor(**params)
            score = cross_val_score(model, Xtr, ytr, cv=3).mean()
            v = trial.number + 1; progress.setValue(v); QApplication.processEvents()
            if progress.wasCanceled(): raise optuna.exceptions.TrialPruned()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(obj, n_trials=trials)
        progress.reset()
        self.best_params = study.best_params
        self.btn_train_best.setEnabled(True)
        self.text_model_res.setPlainText(f"Best params: {self.best_params}\nBest score: {study.best_value:.4f}")

    def train_best(self):
        if self.train_df is None or not self.best_params: return
        mname = self.combo_model.currentText(); is_clf = 'Classifier' in mname
        params = self.best_params
        if 'XGBoost' in mname:
            model = XGBClassifier(**params, eval_metric='mlogloss') if is_clf else XGBRegressor(**params)
        else:
            model = LGBMClassifier(**params) if is_clf else LGBMRegressor(**params)

        df  = self.train_df.copy(); tgt = self.combo_target.currentText()
        X   = df[self.feature_cols]; y = df[tgt]
        if is_clf:
            y = self.target_encoder.transform(y.astype(str))
        Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
        dlg = QProgressDialog('Training with best params...', None, 0, 0, self)
        dlg.setWindowModality(Qt.WindowModal); dlg.show(); QApplication.processEvents()
        try:
            model.fit(Xtr, ytr); dlg.reset()
            yt = model.predict(Xtr); yv_pred = model.predict(Xv)
            s = f"Model (best params): {mname}\nBest params: {params}\n"
            if is_clf:
                s += (
                    f"Train acc: {accuracy_score(ytr, yt):.4f}\n"
                    f"Val acc: {accuracy_score(yv, yv_pred):.4f}\n"
                    f"Prec(w): {precision_score(yv, yv_pred, average='weighted'):.4f}\n"
                    f"Rec(w): {recall_score(yv, yv_pred, average='weighted'):.4f}\n"
                    f"CM:\n{confusion_matrix(yv, yv_pred)}\n"
                )
            else:
                r2t = model.score(Xtr, ytr); r2v = model.score(Xv, yv_pred)
                mae = mean_absolute_error(yv, yv_pred); mse = mean_squared_error(yv, yv_pred)
                rmse = np.sqrt(mse)
                s += (
                    f"Train R2: {r2t:.4f}\n"
                    f"Val R2: {r2v:.4f}\n"
                    f"MAE: {mae:.4f}\n"
                    f"MSE: {mse:.4f}\n"
                    f"RMSE: {rmse:.4f}\n"
                )
            cv = cross_val_score(model, Xtr, ytr, cv=5)
            s += f"CV mean: {cv.mean():.4f} (std {cv.std():.4f})"
            self.model = model
            self.text_model_res.setPlainText(s)
            self.btn_save_model.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self,'Train Best Error',str(e)); traceback.print_exc()

    def save_model(self):
        if not self.model:
            QMessageBox.warning(self,'Save Model','No model to save'); return
        fn, _ = QFileDialog.getSaveFileName(self,'Save Model','','Pickle Files (*.pkl);;All Files (*)')
        if fn:
            try:
                with open(fn,'wb') as f:
                    pickle.dump(self.model, f)
                QMessageBox.information(self,'Save Model',f'Model saved to {fn}')
            except Exception as e:
                QMessageBox.critical(self,'Save Model Error',str(e))

    # --- Submission ---
    def create_submission_tab(self):
        t = QWidget(); l = QVBoxLayout()
        l.addWidget(QLabel('Submission'))
        h = QHBoxLayout(); h.addWidget(QLabel('ID column:'))
        self.combo_id = QComboBox(); h.addWidget(self.combo_id); l.addLayout(h)
        btn = QPushButton('Generate Submission'); btn.clicked.connect(self.generate_submission)
        l.addWidget(btn); t.setLayout(l); self.tabs.addTab(t,'Submission')

    def generate_submission(self):
        if not self.model or self.test_df is None:
            QMessageBox.warning(self,'Submission','Train model and load test data first'); return

        idc   = self.combo_id.currentText()
        df    = self.test_df.copy()
        subdf = self.submission_df.copy()

        if idc not in subdf.columns:
            QMessageBox.warning(self,'Submission','Invalid ID column'); return

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            QMessageBox.warning(self,'Submission',f'Missing columns in test: {missing}'); return

        X     = df[self.feature_cols]
        preds = self.model.predict(X)
        if self.target_encoder:
            try:
                preds = self.target_encoder.inverse_transform(preds)
            except:
                pass

        sub = pd.DataFrame({ idc: subdf[idc], 'prediction': preds })
        fn, _ = QFileDialog.getSaveFileName(self,'Save Submission','submission.csv','CSV Files (*.csv)')
        if fn:
            try:
                sub.to_csv(fn, index=False)
                QMessageBox.information(self,'Submission',f'Saved {fn}')
            except Exception as e:
                QMessageBox.critical(self,'Submission Error',str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w   = DataAnalysisTool()
    sys.exit(app.exec_())
