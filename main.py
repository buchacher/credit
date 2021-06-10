import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- Part 1: Data Processing ---

# To load data, place crx.data in the root directory
df_initial = pd.read_csv('crx.data', header=None)

# Setup for data transformation according to crx.names
target_idx = 15
binary_features_idx = [8, 9, 11]
nominal_features_idx = [0, 3, 4, 5, 6, 12]
continuous_features_idx = [1, 2, 7, 10, 13, 14]


def inspect_non_continuous(data_frame, binary_idx, nominal_idx):
    """Inspect unique values by column, excluding continuous features."""
    for i in range(len(data_frame.columns)):
        if i in binary_idx or i in nominal_idx:
            print("A" + str(i + 1) + ": " + str(data_frame[i].unique()))


# Uncomment below line to output unique values by non-continuous feature
# inspect_non_continuous(df_initial, nominal_features_idx, binary_features_idx)


"""It was verified that '?' values correspond to missing values according to
names.crx. Accordingly, the data is reloaded, passing na_values='?'.
"""
df = pd.read_csv('crx.data', header=None, na_values='?')

# Encode target: assign 0 if outcome is '-' and 1 if '+'
df[target_idx] = [0 if x == '-' else 1 for x in df[target_idx]]

# Split data into input X and output y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode binary features: 0 if value is 'f' and 1 if 't')
for i in binary_features_idx:
    X[i] = [0 if x == 'f' else 1 if x == 't' else np.nan for x in X[i]]

# One-hot encode nominal features only
ct_ohe = ColumnTransformer(
    transformers=[('nom', OneHotEncoder(), nominal_features_idx)],
    remainder='passthrough'
)
X = ct_ohe.fit_transform(X)

# Perform train-test split early in order to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


def prepare_input_data(feature_vector):
    """Prepare input data. Intended to enable preparation of training and test
    input data at different points in order to avoid data leakage.
    """
    # Impute missing values
    simple_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    feature_vector = simple_imputer.fit_transform(feature_vector)

    # Standardise features
    std_scaler = StandardScaler()
    feature_vector = std_scaler.fit_transform(feature_vector)

    # Transform back to pandas DataFrame and return
    return pd.DataFrame(feature_vector)


# --- Part 2: Training ---

# Prepare training input data
X_train_prepared = prepare_input_data(X_train)

# Train the model
log_reg = LogisticRegression(
    penalty='none',
    class_weight='balanced',
    max_iter=5000
)
log_reg.fit(X_train_prepared, y_train)

# --- Part 3: Evaluation ---

# Evaluate model performance on the training set using 3-fold cross validation
train_accuracy = cross_val_score(log_reg, X_train_prepared, y_train, cv=3)
print("Training set CV mean accuracy: %.3f" % train_accuracy.mean())


def prepare_evaluate_test():
    """Prepare test input data and evaluate model performance on the test set.
    Intended to be called as the final step in order avoid data leakage.
    This method call should constitute the final line of the program.
    """
    # Prepare test input data
    X_test_prepared = prepare_input_data(X_test)
    # Predict confidence scores
    y_score = log_reg.decision_function(X_test_prepared)
    # Predict class labels
    y_hat = log_reg.predict(X_test_prepared)

    # 3(a) classification accuracy and 3(b) balanced accuracy
    test_class_accuracy = metrics.accuracy_score(y_test, y_hat)
    test_bal_accuracy = metrics.balanced_accuracy_score(y_test, y_hat)
    print("Test set classification accuracy: %.3f" % test_class_accuracy)
    print("Test set balanced accuracy: %.3f" % test_bal_accuracy)

    # 3(c) Confusion matrix (dependent on class_weight param passed to log_reg)
    conf_matrix = confusion_matrix(y_test, y_hat)
    print(conf_matrix)

    # 3(d) Precision-recall-curve and Average Precision
    test_ap = average_precision_score(y_test, y_score)
    disp = plot_precision_recall_curve(log_reg, X_test_prepared, y_test)
    disp.ax_.set_title('Credit Approval Precision-Recall Curve: AP=%.3f}'
                       % test_ap)
    disp.plot()
    plt.show()
    print("Test set AP: %.3f" % test_ap)


prepare_evaluate_test()
