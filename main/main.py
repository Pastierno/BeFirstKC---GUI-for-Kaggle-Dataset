import sys, io, os, pandas as pd, numpy as np, optuna, seaborn as sns, traceback, pickle, random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QComboBox, QScrollArea,
    QGroupBox, QLabel, QMessageBox, QListWidget, QSplitter, QSpinBox,
    QProgressDialog, QTableWidget, QTableWidgetItem, QPlainTextEdit, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder, OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from PyQt5.QtGui import QFontDatabase, QFont
from widgets import AnimatedButton

class DataAnalysisTool(QMainWindow): # Classe principale
    def __init__(self):
        super().__init__()
        self.train_df = None
        self.test_df = None
        self.submission_df = None
        self.target_encoder = None
        self.model = None
        self.best_params = None
        self.feature_cols = None
        self.current_figsize = (10, 8)
        self.init_ui() # Inizializza l'interfaccia utente

    def init_ui(self):
        self.setWindowTitle('BeFirstKC') # Titolo della finestra
        self.setGeometry(100, 100, 1300, 900) # Dimensioni della finestra
        self.tabs = QTabWidget()
        self.create_load_tab()
        self.create_preprocess_tab()
        self.create_cat_impute_tab()
        self.create_encode_tab()
        self.create_visualize_tab()
        self.create_model_tab()
        self.create_submission_tab()
        self.setCentralWidget(self.tabs)

        # Disabilito tutte le tab finché non scelgo il target
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) not in ['Load Data', 'Model']:
                self.tabs.setTabEnabled(i, False)
        self.combo_target.currentIndexChanged.connect(self.check_enable_tabs)

        self.show()

    def check_enable_tabs(self, idx): # Controlla se il target è selezionato
        if idx > 0:  # > 0 significa che ha selezionato un target reale
            for i in range(self.tabs.count()):
                self.tabs.setTabEnabled(i, True)

    # Carica i dati
    def create_load_tab(self):
        t = QWidget(); l = QVBoxLayout()
         # Crea il label con il messaggio
        label = QLabel(
            "Please select the target column first in the Model tab."
        )
        # Allinea al centro
        label.setAlignment(Qt.AlignCenter)
        # Lo aggiunge al layout
        l.addWidget(label)
        h = QHBoxLayout() # Layout per il caricamento dei file
        self.btn_load_train = QPushButton('Load Train Dataset')
        self.btn_load_test  = QPushButton('Load Test Dataset')
        self.btn_load_train.clicked.connect(lambda: self.load_file('train'))
        self.btn_load_test.clicked.connect(lambda: self.load_file('test'))
        h.addWidget(self.btn_load_train); h.addWidget(self.btn_load_test)
        h.addStretch()
        h.addWidget(QLabel('Preview:'))
        self.combo_head_ds = QComboBox(); self.combo_head_ds.addItems(['Train','Test'])
        self.combo_head_ds.currentIndexChanged.connect(self.update_head_view)
        h.addWidget(self.combo_head_ds)
        l.addLayout(h)

        h2 = QHBoxLayout() # Layout per il numero di righe da mostrare
        h2.addWidget(QLabel('Rows to show:'))
        self.spin_head_rows = QSpinBox(); self.spin_head_rows.setRange(1, 1000); self.spin_head_rows.setValue(5)
        self.spin_head_rows.valueChanged.connect(self.update_head_view)
        h2.addWidget(self.spin_head_rows); h2.addStretch()
        l.addLayout(h2)

        self.table_head = QTableWidget() # Tabella per mostrare i dati
        self.table_head.setEditTriggers(QTableWidget.NoEditTriggers)
        l.addWidget(self.table_head)

        l.addWidget(QLabel('Statistics (numeric columns):')) # Tabella per le statistiche
        self.table_describe = QTableWidget()
        self.table_describe.setEditTriggers(QTableWidget.NoEditTriggers)
        l.addWidget(self.table_describe)

        t.setLayout(l)
        self.tabs.addTab(t, 'Load Data')

    # Load train/test data
    def load_file(self, which):
        fn, _ = QFileDialog.getOpenFileName(
            self, f'Load {which.title()} Dataset', '',
            'CSV Files (*.csv);;All Files (*)'
        )
        if not fn: return
        try:
            df = pd.read_csv(fn)
            if which == 'train':
                self.train_df = df.copy()
            else:
                self.test_df = df.copy()
                self.submission_df = df.copy()
            self.update_all_lists()
            self.update_head_view()
            QMessageBox.information(self, 'Success', f'{which.title()} dataset loaded successfully')
        except Exception as e:
            QMessageBox.critical(self, 'Load Error', str(e))
            traceback.print_exc()

    # Metodo per il caricamento del file di submission
    def update_head_view(self):
        ds = self.combo_head_ds.currentText().lower()
        df = getattr(self, f"{ds}_df")
        if df is None:
            self.table_head.clear(); self.table_describe.clear(); return
        n = self.spin_head_rows.value()
        head = df.head(n)
        self.table_head.setRowCount(len(head)); self.table_head.setColumnCount(len(head.columns))
        self.table_head.setHorizontalHeaderLabels(list(head.columns))
        for i, row in enumerate(head.itertuples(index=False)):
            for j, val in enumerate(row):
                self.table_head.setItem(i, j, QTableWidgetItem(str(val)))
        self.table_head.resizeColumnsToContents()

        desc = df.select_dtypes(include=np.number).describe()
        if desc.empty:
            self.table_describe.clear()
        else:
            self.table_describe.setRowCount(len(desc)); self.table_describe.setColumnCount(len(desc.columns))
            self.table_describe.setHorizontalHeaderLabels(list(desc.columns))
            self.table_describe.setVerticalHeaderLabels(list(desc.index.astype(str)))
            for i, stat in enumerate(desc.index):
                for j, col in enumerate(desc.columns):
                    self.table_describe.setItem(i, j, QTableWidgetItem(f"{desc.at[stat, col]:.4f}"))
            self.table_describe.resizeColumnsToContents()

    # Metodo per il caricamento del file di submission
    def create_preprocess_tab(self):
        t = QWidget(); l = QVBoxLayout()
        ds = QHBoxLayout(); ds.addWidget(QLabel('Dataset:'))
        self.combo_pp_ds = QComboBox(); self.combo_pp_ds.addItems(['Train','Test','Both']) # ComboBox per la selezione del dataset (anche entrambi)
        self.combo_pp_ds.currentIndexChanged.connect(self.update_pp_info)
        ds.addWidget(self.combo_pp_ds); l.addLayout(ds)

        sp = QSplitter(Qt.Horizontal) # Layout per la selezione delle colonne
        sw = QWidget(); sl = QVBoxLayout()
        sa = QPushButton('Select All'); da = QPushButton('Deselect All')
        sa.clicked.connect(lambda: self.list_pp_cols.selectAll())
        da.clicked.connect(lambda: self.list_pp_cols.clearSelection())
        sl.addWidget(sa); sl.addWidget(da)
        self.list_pp_cols = QListWidget(); self.list_pp_cols.setSelectionMode(QListWidget.MultiSelection)
        sl.addWidget(self.list_pp_cols); sw.setLayout(sl)

        tg = QGroupBox('Transforms'); tl = QVBoxLayout() # Layout per le trasformazioni
        yb = QPushButton('Yeo-Johnson'); ss = QPushButton('Standard Scaling') # Pulsanti per le trasformazioni (Yeo-Johnson e Standard Scaling)
        yb.clicked.connect(self.apply_yeo); ss.clicked.connect(self.apply_std)
        tl.addWidget(yb); tl.addWidget(ss); tg.setLayout(tl)

        sp.addWidget(sw); sp.addWidget(tg); l.addWidget(sp)

        ig = QGroupBox('Info & Missing'); il = QVBoxLayout() # Layout per le informazioni e i valori mancanti
        self.text_info = QPlainTextEdit(); self.text_info.setReadOnly(True)
        self.text_nan  = QPlainTextEdit(); self.text_nan.setReadOnly(True)
        il.addWidget(QLabel('Info:'));    il.addWidget(self.text_info)
        il.addWidget(QLabel('Missing:')); il.addWidget(self.text_nan)
        ig.setLayout(il); l.addWidget(ig)

        hb = QHBoxLayout() # Layout per le operazioni di preprocessing
        dn = QPushButton('Drop NaN'); dn.clicked.connect(self.pp_dropna)
        im = QComboBox(); im.addItems(['Mean','Median','Mode'])
        ib = QPushButton('Impute Numeric'); ib.clicked.connect(self.pp_impute)
        hb.addWidget(dn); hb.addWidget(im); hb.addWidget(ib)
        self.combo_imp = im; l.addLayout(hb)

        t.setLayout(l); self.tabs.addTab(t,'Preprocess')

    def update_pp_info(self): # Metodo per aggiornare le informazioni sulle colonne
        ds = self.combo_pp_ds.currentText()
        # seleziono df per info, ma elenco colonne in base a "Both" o singolo
        if ds == 'Train':
            df = self.train_df
            numeric_cols = list(df.select_dtypes(include=np.number).columns) if df is not None else []
        elif ds == 'Test':
            df = self.test_df
            numeric_cols = list(df.select_dtypes(include=np.number).columns) if df is not None else []
        else:  # Both: interseco
            if self.train_df is None or self.test_df is None:
                self.list_pp_cols.clear(); self.text_info.clear(); self.text_nan.clear(); return
            df = self.train_df
            cols_train = set(self.train_df.select_dtypes(include=np.number).columns)
            cols_test  = set(self.test_df.select_dtypes(include=np.number).columns)
            numeric_cols = sorted(cols_train & cols_test)

        # rimuove il target
        tgt = self.combo_target.currentText()
        if tgt in numeric_cols:
            numeric_cols.remove(tgt)

        self.list_pp_cols.clear()
        self.list_pp_cols.addItems(numeric_cols)

        # Info & Missing basate su train
        if self.train_df is not None:
            buf = io.StringIO(); self.train_df.info(buf=buf)
            self.text_info.setPlainText(buf.getvalue())
            self.text_nan.setPlainText('\n'.join(f'{c}: {n}' for c,n in self.train_df.isna().sum().items()))
        else:
            self.text_info.clear(); self.text_nan.clear()

    def apply_yeo(self): # Metodo per applicare la trasformazione Yeo-Johnson
        ds = self.combo_pp_ds.currentText()
        sel = [i.text() for i in self.list_pp_cols.selectedItems()]
        cols = [c for c in sel if c != self.combo_target.currentText()]
        if not cols:
            QMessageBox.warning(self, 'Transform Error', 'Nessuna colonna valida selezionata')
            return

        # Sceglie il DataFrame
        if ds == 'Both':
            dfs = []
            if self.train_df is not None:
                dfs.append(('train', self.train_df))
            if self.test_df is not None:
                dfs.append(('test', self.test_df))
        else:
            name = 'train' if ds == 'Train' else 'test'
            df_sel = getattr(self, f"{name}_df")
            dfs = [(name, df_sel)] if df_sel is not None else []

        for name, df in dfs:
            # Rimuove colonne costanti
            valid = [c for c in cols if c in df.columns and df[c].nunique() > 1]
            skipped = set(cols) - set(valid)
            if skipped:
                print(f"[{name}] saltate costanti: {skipped}")

            # Pulisce NaN e infiniti
            df[valid] = df[valid].replace([np.inf, -np.inf], np.nan).fillna(df[valid].median())

            # Applica Yeo–Johnson colonna per colonna
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            for col in valid:
                try:
                    df[[col]] = pt.fit_transform(df[[col]])
                except Exception as e:
                    # intercetta BracketError e altri
                    print(f"[{name}] impossibile trasformare {col}: {e}")

        self.update_pp_info()
        self.update_viz()


    def apply_std(self): # Metodo per applicare lo standard scaling
        ds = self.combo_pp_ds.currentText()
        sel = [i.text() for i in self.list_pp_cols.selectedItems()]
        cols = [c for c in sel if c != self.combo_target.currentText()]
        if not cols:
            QMessageBox.warning(self,'Transform Error','Nessuna colonna valida selezionata'); return

        if ds == 'Both': # Seleziona i DataFrame
            if self.train_df is not None:
                sc1 = StandardScaler() # StandardScaler per il train
                self.train_df[cols] = sc1.fit_transform(self.train_df[cols])
            if self.test_df is not None:
                cols_t = [c for c in cols if c in self.test_df.columns]
                sc2 = StandardScaler() # StandardScaler per il test
                self.test_df[cols_t] = sc2.fit_transform(self.test_df[cols_t])
        else:
            df = self.get_pp_df()
            sc = StandardScaler()
            df[cols] = sc.fit_transform(df[cols])

        self.update_pp_info(); self.update_viz()

    def pp_dropna(self): # Metodo per rimuovere i valori NaN
        ds = self.combo_pp_ds.currentText()
        if ds == 'Both':
            if self.train_df is not None: self.train_df.dropna(inplace=True)
            if self.test_df is not None:  self.test_df.dropna(inplace=True)
        else:
            df = self.get_pp_df()
            if df is not None: df.dropna(inplace=True)
        self.update_pp_info()

    def pp_impute(self): # Metodo per imputare i valori NaN
        ds = self.combo_pp_ds.currentText()
        m = self.combo_imp.currentText()
        targets = []
        if ds in ['Train','Both']:
            targets.append(self.train_df)
        if ds in ['Test','Both']:
            targets.append(self.test_df)

        for df in targets:
            if df is None: continue
            for c in df.select_dtypes(include=np.number).columns:
                if c == self.combo_target.currentText():
                    continue
                if m == 'Mean':
                    df[c] = df[c].fillna(df[c].mean())
                elif m == 'Median':
                    df[c] = df[c].fillna(df[c].median())
                else:
                    df[c] = df[c].fillna(df[c].mode()[0])

        self.update_pp_info()

    def get_pp_df(self):
        return self.train_df if self.combo_pp_ds.currentText()=='Train' else self.test_df

    # Imputazione delle features categoriche
    def create_cat_impute_tab(self):
        t = QWidget(); l = QVBoxLayout()
        ds = QHBoxLayout(); ds.addWidget(QLabel('Dataset:'))
        self.combo_cat_imp_ds = QComboBox(); self.combo_cat_imp_ds.addItems(['Train','Test','Both'])
        self.combo_cat_imp_ds.currentIndexChanged.connect(self.update_cat_imp)
        ds.addWidget(self.combo_cat_imp_ds); l.addLayout(ds)

        # strategy selection
        h2 = QHBoxLayout()
        h2.addWidget(QLabel('Strategy:'))
        self.combo_cat_imp_strategy = QComboBox()
        self.combo_cat_imp_strategy.addItems(['Mode','Constant','Random']) # Modalità di imputazione
        h2.addWidget(self.combo_cat_imp_strategy)
        l.addLayout(h2)

        hb = QHBoxLayout()
        sa = QPushButton('Select All'); da = QPushButton('Deselect All')
        sa.clicked.connect(lambda: self.list_cat_imp_cols.selectAll())
        da.clicked.connect(lambda: self.list_cat_imp_cols.clearSelection())
        hb.addWidget(sa); hb.addWidget(da); l.addLayout(hb)

        self.list_cat_imp_cols = QListWidget(); self.list_cat_imp_cols.setSelectionMode(QListWidget.MultiSelection)
        l.addWidget(QLabel('Categorical Columns:')); l.addWidget(self.list_cat_imp_cols)

        btn = QPushButton('Impute'); btn.clicked.connect(self.apply_cat_impute)
        l.addWidget(btn)

        t.setLayout(l); self.tabs.addTab(t,'Impute Cat')

    def update_cat_imp(self): # Metodo per aggiornare le colonne da imputare
        ds = self.combo_cat_imp_ds.currentText()
        self.list_cat_imp_cols.clear()
        cols = []
        if ds == 'Train':
            if self.train_df is not None:
                cols = self.train_df.select_dtypes(exclude=np.number).columns.tolist()
        elif ds == 'Test':
            if self.test_df is not None:
                cols = self.test_df.select_dtypes(exclude=np.number).columns.tolist()
        else:  # Both
            if self.train_df is not None:
                cols += self.train_df.select_dtypes(exclude=np.number).columns.tolist()
            if self.test_df is not None:
                cols += self.test_df.select_dtypes(exclude=np.number).columns.tolist()
            # Mantiene solo le colonne uniche
            cols = list(dict.fromkeys(cols))
        self.list_cat_imp_cols.addItems(cols)

    def apply_cat_impute(self): # Metodo per imputare le colonne categoriche
        strategy = self.combo_cat_imp_strategy.currentText()
        cols = [i.text() for i in self.list_cat_imp_cols.selectedItems()]
        if not cols:
            QMessageBox.warning(self,'Impute Cat','No columns selected'); return
        ds = self.combo_cat_imp_ds.currentText()
        targets = []
        if ds in ['Train','Both']:
            targets.append(self.train_df)
        if ds in ['Test','Both']:
            targets.append(self.test_df)
        for df in targets:
            if df is None: continue
            for c in cols:
                if strategy == 'Mode':
                    df[c].fillna(df[c].mode()[0], inplace=True)
                elif strategy == 'Constant':
                    df[c].fillna('missing', inplace=True)
                else:  # Random
                    vals = df[c].dropna().tolist()
                    df[c] = df[c].apply(lambda x: random.choice(vals) if pd.isna(x) and vals else x)
        QMessageBox.information(self,'Impute Cat',f'Imputed {cols} using {strategy}')
        self.update_pp_info(); self.update_enc(); self.update_viz(); self.update_all_lists()

    # Encoding
    def create_encode_tab(self): # Metodo per l'encoding delle colonne categoriche
        t = QWidget(); l = QVBoxLayout()
        l.addWidget(QLabel('Dataset:'))
        self.combo_enc_ds = QComboBox(); self.combo_enc_ds.addItems(['Train','Test','Both'])
        self.combo_enc_ds.currentIndexChanged.connect(self.update_enc)
        l.addWidget(self.combo_enc_ds)

        hb = QHBoxLayout()
        sa = QPushButton('Select All'); da = QPushButton('Deselect All')
        sa.clicked.connect(lambda: self.list_enc.selectAll()); da.clicked.connect(lambda: self.list_enc.clearSelection())
        hb.addWidget(sa); hb.addWidget(da); l.addLayout(hb)

        self.list_enc = QListWidget(); self.list_enc.setSelectionMode(QListWidget.MultiSelection)
        l.addWidget(QLabel('Categorical Columns:')); l.addWidget(self.list_enc)

        self.cb_label = QCheckBox('Label Encoding')
        self.cb_onehot = QCheckBox('OneHot Encoding')
        l.addWidget(self.cb_label); l.addWidget(self.cb_onehot)

        btn = QPushButton('Apply Encoding'); btn.clicked.connect(self.apply_enc)
        l.addWidget(btn)

        t.setLayout(l); self.tabs.addTab(t,'Encode')

    def update_enc(self): # Metodo per aggiornare le colonne da codificare
        ds = self.combo_enc_ds.currentText()
        self.list_enc.clear()
        cols = []
        if ds == 'Train':
            if self.train_df is not None:
                cols = self.train_df.select_dtypes(exclude=np.number).columns.tolist()
        elif ds == 'Test':
            if self.test_df is not None:
                cols = self.test_df.select_dtypes(exclude=np.number).columns.tolist()
        else:  # Both
            if self.train_df is not None:
                cols += self.train_df.select_dtypes(exclude=np.number).columns.tolist()
            if self.test_df is not None:
                cols += self.test_df.select_dtypes(exclude=np.number).columns.tolist()
            cols = list(dict.fromkeys(cols))
        self.list_enc.addItems(cols)

    def apply_enc(self): # Metodo per applicare l'encoding
        cols = [i.text() for i in self.list_enc.selectedItems()]
        if not cols:
            QMessageBox.warning(self,'Encode','No columns selected'); return
        ds = self.combo_enc_ds.currentText()
        datasets = []
        if ds in ['Train','Both']:
            datasets.append(('train', self.train_df))
        if ds in ['Test','Both']:
            datasets.append(('test', self.test_df))
        for name, df in datasets:
            if df is None: continue
            if self.cb_label.isChecked():
                le = LabelEncoder()
                for c in cols:
                    if c in df.columns:
                        df[c] = le.fit_transform(df[c].astype(str))
            if self.cb_onehot.isChecked():
                ohe = OneHotEncoder(sparse_output=False, drop='first')
                exist = [c for c in cols if c in df.columns]
                if exist:
                    arr = ohe.fit_transform(df[exist].astype(str))
                    new_cols = ohe.get_feature_names_out(exist)
                    df_ohe = pd.DataFrame(arr, columns=new_cols, index=df.index)
                    df.drop(columns=exist, inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    df = pd.concat([df, df_ohe.reset_index(drop=True)], axis=1)
                    setattr(self, f"{name}_df", df)
        QMessageBox.information(self,'Encode','Applied encoding')
        self.update_enc(); self.update_pp_info(); self.update_all_lists()

    # Visualizzazione
    def create_visualize_tab(self): # Metodo per la visualizzazione dei dati
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

        sp.setSizes([200,800]); l.addWidget(sp)
        t.setLayout(l); self.tabs.addTab(t,'Visualize')

    def update_viz(self): # Metodo per aggiornare le colonne da visualizzare
        df = self.train_df if self.combo_viz_ds.currentText()=='Train' else self.test_df
        self.list_viz_cols.clear()
        if df is not None:
            self.list_viz_cols.addItems(df.select_dtypes(include=np.number).columns)

    def plot_viz(self): # Metodo per visualizzare i dati
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
                sns.heatmap(corr, mask=np.triu(np.ones_like(corr, bool)), ax=ax, cmap='viridis')
                ax.tick_params(axis='both', labelsize=max(6,12-len(cols)))
        except Exception as e:
            QMessageBox.warning(self,'Plot Error',str(e))
        self.canvas.draw()

    # Model e Optuna
    def create_model_tab(self): # Metodo per la creazione del modello
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
        l.addWidget(QLabel('Model:')); self.combo_model = QComboBox()
        self.combo_model.addItems(['XGBoost Classifier','XGBoost Regressor','LightGBM Classifier','LightGBM Regressor']) # Modelli disponibili
        l.addWidget(self.combo_model)

        tb = QPushButton('Train'); tb.clicked.connect(self.train_model); l.addWidget(tb)
        hb2 = QHBoxLayout(); hb2.addWidget(QLabel('Trials:'))
        self.spin_trials = QSpinBox(); self.spin_trials.setRange(1,500); self.spin_trials.setValue(50)
        hb2.addWidget(self.spin_trials)
        op = QPushButton('Optimize'); op.clicked.connect(self.optimize_model); hb2.addWidget(op)
        l.addLayout(hb2)

        self.btn_train_best = QPushButton('Train Best Params'); self.btn_train_best.setEnabled(False)
        self.btn_train_best.clicked.connect(self.train_best); l.addWidget(self.btn_train_best)
        self.btn_save_model = QPushButton('Save Model'); self.btn_save_model.setEnabled(False)
        self.btn_save_model.clicked.connect(self.save_model); l.addWidget(self.btn_save_model)

        self.text_model_res = QPlainTextEdit(); self.text_model_res.setReadOnly(True); l.addWidget(self.text_model_res)
        t.setLayout(l); self.tabs.addTab(t,'Model')

    def update_all_lists(self): # Metodo per aggiornare tutte le liste
        cols = list(self.train_df.columns) if self.train_df is not None else []
        for w in [self.list_pp_cols, self.list_model_drop]:
            w.clear(); w.addItems(cols)
        # placeholder + target list
        self.combo_target.clear()
        self.combo_target.addItem('Select Target')
        self.combo_target.addItems(cols)
        ids = list(self.submission_df.columns) if self.submission_df is not None else cols
        self.combo_id.clear(); self.combo_id.addItems(ids)
        self.update_pp_info(); self.update_enc(); self.update_viz()

    def model_drop(self): # Metodo per rimuovere le colonne selezionate
        to_drop = [i.text() for i in self.list_model_drop.selectedItems()]
        if not to_drop:
            QMessageBox.warning(self,'Model','No columns selected'); return
        if self.train_df is not None:
            self.train_df.drop(columns=to_drop, inplace=True, errors='ignore')
        if self.test_df is not None:
            self.test_df.drop(columns=to_drop, inplace=True, errors='ignore')
        self.update_all_lists()
        QMessageBox.information(self,'Model',f'Dropped {to_drop}')

    def train_model(self): # Metodo per addestrare il modello
        if self.train_df is None:
            QMessageBox.warning(self,'Model','Load train data first'); return
        df = self.train_df.copy(); tgt = self.combo_target.currentText()
        if tgt == 'Select Target' or tgt not in df.columns:
            QMessageBox.warning(self,'Model','Select a valid target'); return
        X = df.drop(columns=[tgt]).select_dtypes(include=np.number); y = df[tgt]
        is_clf = 'Classifier' in self.combo_model.currentText()
        if is_clf:
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y.astype(str))
            mdl = XGBClassifier(eval_metric='mlogloss') if 'XGBoost' in self.combo_model.currentText() else LGBMClassifier()
        else:
            base = XGBRegressor() if 'XGBoost' in self.combo_model.currentText() else LGBMRegressor()
            mdl = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
        self.model = mdl; self.feature_cols = X.columns.tolist()
        Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
        dlg = QProgressDialog('Training...', None, 0, 0, self); dlg.setWindowModality(Qt.WindowModal); dlg.show()
        QApplication.processEvents()
        try:
            self.model.fit(Xtr, ytr); dlg.reset()
            yt, yv_pred = self.model.predict(Xtr), self.model.predict(Xv)
            s = f"Model: {self.combo_model.currentText()}\n"
            if is_clf:
                s += (
                    f"Train acc: {accuracy_score(ytr, yt):.4f}\n"
                    f"Val acc: {accuracy_score(yv, yv_pred):.4f}\n"
                    f"Prec(w): {precision_score(yv, yv_pred, average='weighted'):.4f}\n"
                    f"Rec(w): {recall_score(yv, yv_pred, average='weighted'):.4f}\n"
                    f"CM:\n{confusion_matrix(yv, yv_pred)}\n"
                )
            else:
                mae = mean_absolute_error(yv, yv_pred); mse = mean_squared_error(yv, yv_pred)
                rmse = np.sqrt(mse)
                s += (
                    f"Train R2: {self.model.score(Xtr, ytr):.4f}\n"
                    f"Val R2: {self.model.score(Xv, yv_pred):.4f}\n"
                    f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\n"
                )
            cv = cross_val_score(self.model, Xtr, ytr, cv=5)
            s += f"CV mean: {cv.mean():.4f} (std {cv.std():.4f})"
            self.text_model_res.setPlainText(s)
            self.btn_save_model.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self,'Train Error',str(e)); traceback.print_exc()

    def optimize_model(self): # Metodo per ottimizzare il modello
        if self.train_df is None:
            QMessageBox.warning(self,'Model','Load train data first'); return
        df = self.train_df.copy(); tgt = self.combo_target.currentText()
        if tgt == 'Select Target' or tgt not in df.columns:
            QMessageBox.warning(self,'Model','Select a valid target'); return
        X = df.drop(columns=[tgt]).select_dtypes(include=np.number); y = df[tgt]
        if 'Classifier' in self.combo_model.currentText() and self.target_encoder:
            y = self.target_encoder.transform(y.astype(str))
        self.feature_cols = X.columns.tolist()
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        trials = self.spin_trials.value(); mname = self.combo_model.currentText()
        progress = QProgressDialog('Optuna...', 'Cancel', 0, trials, self); progress.setWindowModality(Qt.WindowModal); progress.show()

        def obj(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }
            if 'XGBoost' in mname:
                mdl = XGBClassifier(**params, eval_metric='mlogloss') if 'Classifier' in mname else XGBRegressor(**params)
            else:
                mdl = LGBMClassifier(**params) if 'Classifier' in mname else LGBMRegressor(**params)
            return cross_val_score(mdl, Xtr, ytr, cv=3).mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(obj, n_trials=trials)
        progress.reset()
        self.best_params = study.best_params
        self.btn_train_best.setEnabled(True)
        self.text_model_res.setPlainText(f"Best params: {self.best_params}\nBest score: {study.best_value:.4f}")

    def train_best(self): # Metodo per addestrare il miglior modello
        if self.train_df is None or not self.best_params: return
        is_clf = 'Classifier' in self.combo_model.currentText()
        params = self.best_params
        if is_clf:
            mdl = XGBClassifier(**params, eval_metric='mlogloss') if 'XGBoost' in self.combo_model.currentText() else LGBMClassifier(**params)
        else:
            base = XGBRegressor(**params) if 'XGBoost' in self.combo_model.currentText() else LGBMRegressor(**params)
            mdl = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
        self.model = mdl
        df = self.train_df.copy(); tgt = self.combo_target.currentText()
        X = df[self.feature_cols]; y = df[tgt]
        if is_clf: y = self.target_encoder.transform(y.astype(str))
        Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
        dlg = QProgressDialog('Training best...', None, 0, 0, self); dlg.setWindowModality(Qt.WindowModal); dlg.show(); QApplication.processEvents()
        try:
            self.model.fit(Xtr, ytr); dlg.reset()
            yt, yv_pred = self.model.predict(Xtr), self.model.predict(Xv)
            s = f"Model (best): {self.combo_model.currentText()}\nBest params: {params}\n"
            if is_clf:
                s += (
                    f"Train acc: {accuracy_score(ytr, yt):.4f}\n"
                    f"Val acc: {accuracy_score(yv, yv_pred):.4f}\n"
                    f"Prec(w): {precision_score(yv, yv_pred, average='weighted'):.4f}\n"
                    f"Rec(w): {recall_score(yv, yv_pred, average='weighted'):.4f}\n"
                    f"CM:\n{confusion_matrix(yv, yv_pred)}\n"
                )
            else:
                mae = mean_absolute_error(yv, yv_pred); mse = mean_squared_error(yv, yv_pred)
                rmse = np.sqrt(mse)
                s += (
                    f"Train R2: {self.model.score(Xtr, ytr):.4f}\n"
                    f"Val R2: {self.model.score(Xv, yv_pred):.4f}\n"
                    f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\n"
                )
            cv = cross_val_score(self.model, Xtr, ytr, cv=5)
            s += f"CV mean: {cv.mean():.4f} (std {cv.std():.4f})"
            self.text_model_res.setPlainText(s)
            self.btn_save_model.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self,'Train Best Error',str(e)); traceback.print_exc()

    def save_model(self): # Metodo per salvare il modello
        if not self.model:
            QMessageBox.warning(self,'Save Model','No model to save'); return
        fn, _ = QFileDialog.getSaveFileName(self,'Save Model','','Pickle Files (*.pkl);;All Files (*)')
        if fn:
            try:
                with open(fn,'wb') as f: pickle.dump(self.model, f)
                QMessageBox.information(self,'Save Model',f'Model saved to {fn}')
            except Exception as e:
                QMessageBox.critical(self,'Save Error',str(e))

    # Submission
    def create_submission_tab(self):
        t = QWidget(); l = QVBoxLayout()
        l.addWidget(QLabel('Submission'))
        h = QHBoxLayout(); h.addWidget(QLabel('ID column:'))
        self.combo_id = QComboBox(); h.addWidget(self.combo_id); l.addLayout(h)
        btn = QPushButton('Generate Submission'); btn.clicked.connect(self.generate_submission)
        l.addWidget(btn); t.setLayout(l); self.tabs.addTab(t,'Submission')

    def generate_submission(self): # Metodo per generare il file di submission
        if not self.model or self.test_df is None:
            QMessageBox.warning(self,'Submission','Train model and load test data first'); return
        idc = self.combo_id.currentText()
        df = self.test_df.copy()
        subdf = self.submission_df.copy()
        if idc not in subdf.columns:
            QMessageBox.warning(self,'Submission','Invalid ID column'); return
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            QMessageBox.warning(self,'Submission',f'Missing columns: {missing}'); return
        X = df[self.feature_cols]
        preds = self.model.predict(X)
        if self.target_encoder:
            try: preds = self.target_encoder.inverse_transform(preds)
            except: pass
        sub = pd.DataFrame({idc: subdf[idc], 'prediction': preds})
        fn, _ = QFileDialog.getSaveFileName(self,'Save Submission','submission.csv','CSV Files (*.csv)')
        if fn:
            try:
                sub.to_csv(fn, index=False)
                QMessageBox.information(self,'Submission',f'Saved {fn}')
            except Exception as e:
                QMessageBox.critical(self,'Submission Error',str(e))

    def update_all_lists(self): # Metodo per aggiornare tutte le liste
            cols = list(self.train_df.columns) if self.train_df is not None else []
            for w in [self.list_pp_cols, self.list_model_drop]:
                w.clear(); w.addItems(cols)
            self.combo_target.clear()
            self.combo_target.addItem('Select Target')
            self.combo_target.addItems(cols)
            if self.submission_df is not None:
                ids = list(self.submission_df.columns)
            else:
                ids = cols
            self.combo_id.clear(); self.combo_id.addItems(ids)
            self.update_pp_info(); self.update_enc(); self.update_viz()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Registra il font Roboto
    font_path = os.path.join(os.path.dirname(__file__), "fonts", "Roboto-Regular.ttf")
    font_id = QFontDatabase.addApplicationFont(font_path)
    if font_id != -1:
        family = QFontDatabase.applicationFontFamilies(font_id)[0]
        app.setFont(QFont(family))
    else:
        print("⚠️ Roboto non caricato, uso il font di sistema.")

    # Carica e applica lo stylesheet
    qss_path = os.path.join(os.path.dirname(__file__), "style.qss")
    with open(qss_path, 'r', encoding='utf-8') as f:
        app.setStyleSheet(f.read())

    # Istanzia e mostra la finestra principale
    w = DataAnalysisTool()
    w.show()
    sys.exit(app.exec_())

