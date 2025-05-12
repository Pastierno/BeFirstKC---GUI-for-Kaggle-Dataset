import sys
import io
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox, QCheckBox, QListWidget,
                             QTextEdit, QAbstractItemView)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import optuna

class DataAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Data Analysis App')
        self.train_df = None
        self.test_df = None
        self.transformed_df = None
        self.model = None
        self.scaler = None
        self.num_features = []
        self.feature_columns = None
        self.target_col = None
        self._build_ui()
        # Set initial window size to accommodate larger plots
        self.showMaximized()

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout()
        # File selectors
        file_layout = QHBoxLayout()
        self.train_btn = QPushButton('Browse Train CSV')
        self.train_btn.clicked.connect(self.load_train)
        self.train_label = QLabel('Train: None')
        self.test_btn = QPushButton('Browse Test CSV')
        self.test_btn.clicked.connect(self.load_test)
        self.test_label = QLabel('Test: None')
        file_layout.addWidget(self.train_btn)
        file_layout.addWidget(self.train_label)
        file_layout.addWidget(self.test_btn)
        file_layout.addWidget(self.test_label)
        layout.addLayout(file_layout)
        # Target selector
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel('Target column:'))
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        target_layout.addWidget(self.target_combo)
        layout.addLayout(target_layout)
        # Drop columns selector
        drop_layout = QVBoxLayout()
        drop_layout.addWidget(QLabel('Select columns to drop:'))
        self.drop_list = QListWidget()
        self.drop_list.setSelectionMode(QAbstractItemView.MultiSelection)
        drop_layout.addWidget(self.drop_list)
        self.drop_btn = QPushButton('Drop Selected Columns')
        self.drop_btn.clicked.connect(self.apply_drop_cols)
        drop_layout.addWidget(self.drop_btn)
        layout.addLayout(drop_layout)
        # Dtype conversion selectors
        dtype_layout = QVBoxLayout()
        dtype_layout.addWidget(QLabel('Select columns to convert dtype:'))
        self.dtype_list = QListWidget()
        self.dtype_list.setSelectionMode(QAbstractItemView.MultiSelection)
        dtype_layout.addWidget(self.dtype_list)
        type_choice_layout = QHBoxLayout()
        type_choice_layout.addWidget(QLabel('To dtype:'))
        self.dtype_type_combo = QComboBox()
        self.dtype_type_combo.addItems(['int', 'float', 'str', 'category'])
        type_choice_layout.addWidget(self.dtype_type_combo)
        dtype_layout.addLayout(type_choice_layout)
        self.dtype_btn = QPushButton('Apply Dtype')
        self.dtype_btn.clicked.connect(self.apply_dtype)
        dtype_layout.addWidget(self.dtype_btn)
        layout.addLayout(dtype_layout)
        # Log transform selectors
        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel('Select numeric columns to log-transform:'))
        self.log_list = QListWidget()
        self.log_list.setSelectionMode(QAbstractItemView.MultiSelection)
        log_layout.addWidget(self.log_list)
        self.log_btn = QPushButton('Apply Log Transform')
        self.log_btn.clicked.connect(self.apply_log)
        log_layout.addWidget(self.log_btn)
        layout.addLayout(log_layout)
        # Null handling
        null_layout = QHBoxLayout()
        null_layout.addWidget(QLabel('Nulls:'))
        self.null_combo = QComboBox()
        self.null_combo.addItems(['None','Drop Nulls','Fill Mean','Fill Median','Fill Mode'])
        null_layout.addWidget(self.null_combo)
        layout.addLayout(null_layout)
        # Duplicates
        dup_layout = QHBoxLayout()
        self.dup_check = QCheckBox('Drop Duplicates')
        dup_layout.addWidget(self.dup_check)
        layout.addLayout(dup_layout)
        # Standardize
        std_layout = QHBoxLayout()
        self.std_check = QCheckBox('Standardize')
        std_layout.addWidget(self.std_check)
        layout.addLayout(std_layout)
                        # Visualization options
        viz_layout = QHBoxLayout()
        self.hist_check = QCheckBox('Histplot')
        self.box_check = QCheckBox('Boxplot')
        self.corr_check = QCheckBox('Corr Matrix')
        self.line_check = QCheckBox('Lineplot')
        self.bar_check = QCheckBox('Barplot')
        viz_layout.addWidget(self.hist_check)
        viz_layout.addWidget(self.box_check)
        viz_layout.addWidget(self.corr_check)
        viz_layout.addWidget(self.line_check)
        viz_layout.addWidget(self.bar_check)
        self.viz_btn = QPushButton('Plot')
        self.viz_btn.clicked.connect(self.plot)
        viz_layout.addWidget(self.viz_btn)
        layout.addLayout(viz_layout)

        

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel('Model:'))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            'LogisticRegression','RandomForestClassifier','XGBClassifier','LGBMClassifier',
            'LinearRegression','RandomForestRegressor','XGBRegressor','LGBMRegressor'
        ])
        model_layout.addWidget(self.model_combo)
        self.tune_chk = QCheckBox('Use Optuna Tuning')
        model_layout.addWidget(self.tune_chk)
        self.train_model_btn = QPushButton('Train Model')
        self.train_model_btn.clicked.connect(self.train_model)
        model_layout.addWidget(self.train_model_btn)
        layout.addLayout(model_layout)
        # Predict button
        pred_layout = QHBoxLayout()
        self.pred_btn = QPushButton('Predict Test & Save')
        self.pred_btn.clicked.connect(self.predict)
        pred_layout.addWidget(self.pred_btn)
        layout.addLayout(pred_layout)
        # Output text
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)
        central.setLayout(layout)
        self.setCentralWidget(central)
    def load_train(self):
        path,_=QFileDialog.getOpenFileName(self,'Open Train CSV','','CSV Files (*.csv)')
        if path:
            self.train_df=pd.read_csv(path)
            self.transformed_df=self.train_df.copy()
            self.train_label.setText(f'Train: {path}')
            cols=list(self.train_df.columns)
            self.target_combo.clear();self.target_combo.addItems(cols);self.target_combo.setEnabled(True)
            self.drop_list.clear();self.drop_list.addItems(cols)
            self.dtype_list.clear();self.dtype_list.addItems(cols)
            num_cols=list(self.train_df.select_dtypes(include=np.number).columns)
            self.log_list.clear();self.log_list.addItems(num_cols)
    def load_test(self):
        path,_=QFileDialog.getOpenFileName(self,'Open Test CSV','','CSV Files (*.csv)')
        if path:
            self.test_df=pd.read_csv(path)
            self.test_label.setText(f'Test: {path}')
    def apply_drop_cols(self):
        cols=[i.text() for i in self.drop_list.selectedItems()]
        if self.transformed_df is not None and cols:
            self.transformed_df.drop(columns=cols,inplace=True,errors='ignore')
            self.output.append(f'Dropped columns: {cols}')
            rem=list(self.transformed_df.columns)
            self.drop_list.clear();self.drop_list.addItems(rem)
            self.dtype_list.clear();self.dtype_list.addItems(rem)
            num_cols=list(self.transformed_df.select_dtypes(include=np.number).columns)
            self.log_list.clear();self.log_list.addItems(num_cols)
    def apply_dtype(self):
        cols=[i.text() for i in self.dtype_list.selectedItems()]
        dtype=self.dtype_type_combo.currentText()
        if self.transformed_df is not None and cols:
            for c in cols: self.transformed_df[c]=self.transformed_df[c].astype(dtype)
            self.output.append(f'Converted {cols} to {dtype}')
    def apply_log(self):
        cols=[i.text() for i in self.log_list.selectedItems()]
        if self.transformed_df is not None and cols:
            for c in cols: self.transformed_df[c]=np.log1p(self.transformed_df[c])
            self.output.append(f'Applied log to {cols}')
    def preprocess(self):
        df = self.transformed_df
        method = self.null_combo.currentText()
        # Handle numeric nulls
        if method == 'Drop Nulls':
            df = df.dropna()
        elif method == 'Fill Mean':
            num = df.select_dtypes(include=np.number)
            df[num.columns] = num.fillna(num.mean())
        elif method == 'Fill Median':
            num = df.select_dtypes(include=np.number)
            df[num.columns] = num.fillna(num.median())
        elif method == 'Fill Mode':
            # Fill all with mode
            df = df.fillna(df.mode().iloc[0])
        # Fill categorical NaNs with mode
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
        # Duplicates
        if self.dup_check.isChecked():
            df = df.drop_duplicates()
        # Standardize numeric features
        if self.std_check.isChecked():
            self.scaler = StandardScaler()
            num = df.select_dtypes(include=np.number)
            self.num_features = list(num.columns)
            df[self.num_features] = self.scaler.fit_transform(num)
        self.transformed_df = df
    def plot(self):
        # Preprocess data
        self.preprocess()
        df = self.transformed_df.select_dtypes(include=np.number)
        # Create a separate window for plots
        plot_window = QMainWindow(self)
        plot_window.setWindowTitle('Data Plots')
        # Create figure and canvas
        fig = Figure(figsize=(12, 8))
        canvas = FigureCanvas(fig)
        plot_window.setCentralWidget(canvas)
        # Generate plot
        ax = fig.add_subplot(111)
        if self.hist_check.isChecked():
            df.plot(kind='hist', ax=ax)
        if self.box_check.isChecked():
            df.plot(kind='box', ax=ax)
        if self.corr_check.isChecked():
            import seaborn as sns
            sns.heatmap(df.corr(), annot=True, ax=ax)
        if self.line_check.isChecked():
            df.plot(kind='line', ax=ax)
        if self.bar_check.isChecked():
            # For barplot, plot column means
            df.mean().plot(kind='bar', ax=ax)
        # Show the new plot window
        canvas.draw()
        plot_window.show()
        canvas.draw()
        plot_window.show()
    def optimize_xgb(self,X,y,obj):
        def f(t):
            p={'max_depth':t.suggest_int('md',3,10),'learning_rate':t.suggest_loguniform('lr',1e-3,0.3),'n_estimators':t.suggest_int('ne',50,500),'subsample':t.suggest_uniform('ss',0.5,1.0),'colsample_bytree':t.suggest_uniform('cb',0.5,1.0)}
            m=XGBClassifier(**p,use_label_encoder=False,eval_metric='logloss') if obj=='classifier' else XGBRegressor(**p)
            m.fit(X,y);pr=m.predict(X)
            from sklearn.metrics import accuracy_score,mean_squared_error
            return (1-accuracy_score(y,pr)) if obj=='classifier' else mean_squared_error(y,pr)
        s=optuna.create_study(direction='minimize');s.optimize(f,n_trials=20)
        return s.best_params
    def optimize_lgbm(self,X,y,obj):
        def f(t):
            p={'num_leaves':t.suggest_int('nl',31,256),'learning_rate':t.suggest_loguniform('lr',1e-3,0.3),'n_estimators':t.suggest_int('ne',50,500),'min_child_samples':t.suggest_int('mcs',5,100)}
            m=LGBMClassifier(**p) if obj=='classifier' else LGBMRegressor(**p)
            m.fit(X,y);pr=m.predict(X)
            from sklearn.metrics import accuracy_score,mean_squared_error
            return (1-accuracy_score(y,pr)) if obj=='classifier' else mean_squared_error(y,pr)
        s=optuna.create_study(direction='minimize');s.optimize(f,n_trials=20)
        return s.best_params
    def train_model(self):
        if self.transformed_df is None: return
        self.target_col=self.target_combo.currentText();self.preprocess()
        X=self.transformed_df.drop(self.target_col,axis=1);y=self.transformed_df[self.target_col]
        Xp=pd.get_dummies(X);self.feature_columns=Xp.columns
        mn=self.model_combo.currentText();obj='classifier' if 'Classifier' in mn else 'regressor';p={}
        if self.tune_chk.isChecked():
            if 'XGB' in mn: p=self.optimize_xgb(Xp,y,obj)
            elif 'LGBM' in mn: p=self.optimize_lgbm(Xp,y,obj)
        mdl={'LogisticRegression':LogisticRegression,'RandomForestClassifier':RandomForestClassifier,'XGBClassifier':lambda:XGBClassifier(**p,use_label_encoder=False,eval_metric='logloss') if p else XGBClassifier(use_label_encoder=False,eval_metric='logloss'),'LGBMClassifier':lambda:LGBMClassifier(**p) if p else LGBMClassifier(),'LinearRegression':LinearRegression,'RandomForestRegressor':RandomForestRegressor,'XGBRegressor':lambda:XGBRegressor(**p) if p else XGBRegressor(),'LGBMRegressor':lambda:LGBMRegressor(**p) if p else LGBMRegressor()}[mn]()
        self.model=mdl;self.model.fit(Xp,y);self.output.append(f'Trained {mn} on {self.target_col}')
    def predict(self):
        if self.model is None or self.test_df is None or self.feature_columns is None: return
        df=self.test_df.copy()
        if self.std_check.isChecked():
            for f in self.num_features:
                if f not in df: df[f]=0
            df[self.num_features]=self.scaler.transform(df[self.num_features])
        dp=pd.get_dummies(df);dp=dp.reindex(columns=self.feature_columns,fill_value=0)
        pr=self.model.predict(dp);df['prediction']=pr
        if 'id' in df.columns and self.target_col:
            sub=pd.DataFrame({'id':df['id'],self.target_col:df['prediction']});path,_=QFileDialog.getSaveFileName(self,'Save Submission','submission.csv','CSV Files (*.csv)')
            if path:sub.to_csv(path,index=False);self.output.append(f'Saved: {path}')
        self.output.append(str(df.head()))
if __name__=='__main__':
    app=QApplication(sys.argv);win=DataAnalysisApp();win.show();sys.exit(app.exec_())
