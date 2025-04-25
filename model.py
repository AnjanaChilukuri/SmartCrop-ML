import pickle
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')
sns.set_style("whitegrid", {'axes.grid': False})


def explore_data(df):
    print("Number of Instances and Attributes:", df.shape)
    print('\nDataset columns:', df.columns)
    print('\nData types of each column:')
    print(df.info())


def checking_removing_duplicates(df):
    count_dups = df.duplicated().sum()
    print("Number of Duplicates: ", count_dups)
    if count_dups >= 1:
        df.drop_duplicates(inplace=True)
        print('Duplicate values removed!')
    else:
        print('No Duplicate values')


def read_in_and_split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def GetModel():
    Models = []
    Models.append(('LR', make_pipeline(StandardScaler(), LogisticRegression())))
    Models.append(('LDA', make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())))
    Models.append(('KNN', make_pipeline(StandardScaler(), KNeighborsClassifier())))
    Models.append(('CART', make_pipeline(StandardScaler(), DecisionTreeClassifier())))
    Models.append(('NB', make_pipeline(StandardScaler(), GaussianNB())))
    Models.append(('SVM', make_pipeline(StandardScaler(), SVC(probability=True))))
    Models.append(('RF', make_pipeline(StandardScaler(), RandomForestClassifier())))
    Models.append(('AdaBoost', make_pipeline(StandardScaler(), AdaBoostClassifier())))
    Models.append(('Bagging', make_pipeline(StandardScaler(), BaggingClassifier())))
    Models.append(('ExtraTrees', make_pipeline(StandardScaler(), ExtraTreesClassifier())))
    Models.append(('GradientBoosting', make_pipeline(StandardScaler(), GradientBoostingClassifier())))
    return Models

def remove_outliers(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    return df


def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = []
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        accuracy = model.score(X_test, y_test)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        results.append({
            'Model': name,
            'Test Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
        if y_prob is not None:
            y_bin = label_binarize(y_test, classes=model.classes_)
            n_classes = y_bin.shape[1]
            fpr_dict[name], tpr_dict[name], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
            auc_dict[name] = auc(fpr_dict[name], tpr_dict[name])

    return pd.DataFrame(results), fpr_dict, tpr_dict, auc_dict


def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.show()

def plot_confusion_matrix(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='YlGnBu', fmt='g', cbar=False)
    plt.title(f'Confusion Matrix for {best_model.__class__.__name__}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_roc_curve(fpr_dict, tpr_dict, auc_dict):
    plt.figure(figsize=(10, 8))
    for model_name in fpr_dict:
        plt.plot(fpr_dict[model_name], tpr_dict[model_name], label=f'{model_name} (AUC = {auc_dict[model_name]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Model')
    plt.legend(loc='lower right')
    plt.show()


df = pd.read_csv('SmartCrop-Dataset.csv')
print("Column names:", df.columns)
df.columns = df.columns.str.strip()
numeric_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
print("Unique values in 'label' column:", df['label'].unique())
df.dropna(inplace=True)
df = remove_outliers(df)
target = 'label'
X_train, X_test, y_train, y_test = read_in_and_split_data(df, target)
models = GetModel()
performance_df, fpr_dict, tpr_dict, auc_dict = evaluate_models(models, X_train, y_train, X_test, y_test)
print("\nModel Performance Summary (Table):")
print(performance_df.to_string(index=False))

melted_df = performance_df.melt(id_vars='Model',
                                value_vars=['Test Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                var_name='Metric', value_name='Score')
plt.figure(figsize=(14, 8))
sns.barplot(data=melted_df, x='Model', y='Score', hue='Metric')
plt.title('Model Comparison - Metrics', fontsize=16)
plt.ylabel('Score', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Metric', fontsize=10)
plt.tight_layout()
plt.show()
best_model = models[performance_df['Test Accuracy'].idxmax()][1]
plot_confusion_matrix(best_model, X_test, y_test)
plot_correlation_matrix(df)
plot_roc_curve(fpr_dict, tpr_dict, auc_dict)
pickle.dump(best_model, open('best_model.pkl', 'wb'))
