# 📊 SL-III — Data Science Lab Practicals
### D.Y. Patil College of Engineering | TE | Semester 6

> **Prepared for Viva Examination** — Each practical is explained in full detail covering theory, code logic, concepts, and likely viva questions.

---

## 📁 Directory Structure

```
SL-III/
├── DS01.ipynb   →  Data Preprocessing (Titanic Dataset)
├── DS02.ipynb   →  Data Wrangling (Academic Performance Dataset)
├── DS03.ipynb   →  Descriptive Statistics (Adult Census + Iris)
├── DS04.ipynb   →  Linear Regression (Boston Housing)
├── DS05.ipynb   →  Logistic Regression (Social Network Ads)
├── DS06.ipynb   →  Naive Bayes Classifier (Iris Dataset)
├── DS07.ipynb   →  NLP — Text Preprocessing (NLTK)
├── DS08.ipynb   →  Data Visualization (Titanic Dataset)
├── DS09.ipynb   →  Box Plot Analysis (Titanic — Age vs Gender)
└── DS10.ipynb   →  Feature Analysis with Histograms & Boxplots (Iris)
```

---

## 🔷 DS01 — Data Preprocessing on Titanic Dataset

### 📌 Objective
Perform complete data preprocessing on the Titanic dataset — handling missing values, type conversion, encoding categorical variables, and normalization.

### 📦 Dataset
**Titanic Dataset** — 891 records, 12 columns.  
Downloaded via `kagglehub` (`yasserh/titanic-dataset`).

| Column | Description |
|--------|-------------|
| PassengerId | Unique ID |
| Survived | 0 = No, 1 = Yes |
| Pclass | Ticket class (1, 2, 3) |
| Name, Sex, Age | Personal info |
| SibSp, Parch | Family aboard |
| Ticket, Fare | Ticket details |
| Cabin | Cabin number |
| Embarked | Port (S/C/Q) |

### 🔧 Steps Performed

#### 1. Import Libraries
```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
```

#### 2. Load Dataset via KaggleHub
```python
path = kagglehub.dataset_download("yasserh/titanic-dataset")
df = pd.read_csv(os.path.join(path, "Titanic-Dataset.csv"))
```

#### 3. Exploratory Data Analysis
```python
df.shape        # (891, 12) — rows x columns
df.describe()   # Summary statistics
df.isnull().sum()  # Missing values per column
df.info()       # Data types and non-null counts
```

#### 4. Handle Missing Values
| Column | Missing Count | Strategy Used |
|--------|--------------|---------------|
| Age | 177 | Fill with **mean** |
| Cabin | 687 | Fill with `"Unknown"` |
| Embarked | 2 | Fill with **mode** |

```python
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Cabin'] = df['Cabin'].fillna("Unknown")
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
```

#### 5. Type Conversion
```python
df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
```

#### 6. Label Encoding (Categorical → Numerical)
```python
encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])       # female=0, male=1
df['Embarked'] = encoder.fit_transform(df['Embarked'])  # C=0, Q=1, S=2
```

#### 7. One-Hot Encoding
```python
df = pd.get_dummies(df, columns=['Pclass'])
# Creates: Pclass_1, Pclass_2, Pclass_3 (binary columns)
```

#### 8. Min-Max Normalization
```python
df['Fare_normalized'] = (df['Fare'] - df['Fare'].min()) / (df['Fare'].max() - df['Fare'].min())
```

#### 9. Visualization
```python
sns.countplot(x='Survived', data=df)
```

### 🧠 Key Concepts for Viva

| Concept | Explanation |
|---------|-------------|
| **Missing Value Imputation** | Replacing null values with mean (numerical) or mode (categorical) to avoid data loss |
| **Label Encoding** | Converts text categories to integer codes (female=0, male=1) |
| **One-Hot Encoding** | Creates N binary columns from 1 categorical column with N classes — avoids ordinal assumption |
| **Min-Max Normalization** | Scales values to [0,1] range. Formula: `(x - min) / (max - min)` |
| **`df.describe()`** | Returns count, mean, std, min, 25%, 50%, 75%, max for numeric columns |
| **`df.info()`** | Returns column names, non-null counts, and data types |
| **`astype('category')`** | Efficiently stores repeated string values with numeric codes internally |

### ❓ Likely Viva Questions
1. **Why do we fill missing Age with mean and not mode?**  
   Age is a continuous numerical variable; mean is more representative of the central tendency for numerical data.
2. **What is the difference between Label Encoding and One-Hot Encoding?**  
   Label Encoding assigns integers (may imply order). One-Hot creates binary columns — preferred when no ordinal relationship exists.
3. **Why do we normalize data?**  
   To bring all features to the same scale so no single feature dominates the model (especially important for distance-based algorithms).
4. **What is the Titanic dataset used for?**  
   It's a classic binary classification dataset used to predict survival based on passenger attributes.

---

## 🔷 DS02 — Data Wrangling on Academic Performance Dataset

### 📌 Objective
Perform comprehensive data wrangling on a synthetically generated student academic dataset — fixing inconsistencies, handling missing values, detecting and removing outliers, and applying data transformations.

### 📦 Dataset
**Synthetically generated** using `numpy.random` — 1000 student records.

| Column | Description |
|--------|-------------|
| Student_ID | Unique ID |
| Age | Student age (18–24) |
| Math_Score, Science_Score, English_Score | Exam scores (40–100) |
| Attendance | Attendance % (50–100) |
| Study_Hours | Daily study hours (1–9) |
| Gender | Male/Female (with intentional inconsistencies like 'M', 'F', 'ma') |
| Grade | A/B/C/D/F (with inconsistencies) |

### 🔧 Steps Performed

#### 1. Identify and Fix Categorical Inconsistencies
```python
# Fix Gender: 'M' -> 'Male', 'F' -> 'Female', strip spaces, capitalize
df['Gender'] = df['Gender'].str.strip().str.capitalize()
df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female'})

# Fix Grade: normalize to uppercase
df['Grade'] = df['Grade'].str.strip().str.upper()
```

#### 2. Handle Missing Values
```python
# Numeric columns → fill with median
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns → fill with mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Drop any truly unfixable rows
df.dropna(inplace=True)
```

#### 3. Outlier Detection — IQR Method
```python
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
# Rows outside [lower, upper] are outliers
```

#### 4. Visualize Outliers — Boxplots
```python
sns.boxplot(y=df[col])  # Before removal
# After removal — replot to confirm clean data
```

#### 5. Skewness Analysis
```python
print(df[num_cols].skew())
# Symmetric ≈ 0, Right-skewed > 0, Left-skewed < 0
```

#### 6. Log Transformation (Reduce Skewness)
```python
df['Study_Hours_Log'] = np.log1p(df['Study_Hours'])
```

#### 7. Q-Q Plot (Normality Check)
```python
from scipy import stats
stats.probplot(df['Study_Hours'], dist="norm", plot=axes[0])
stats.probplot(df['Study_Hours_Log'], dist="norm", plot=axes[1])
```

#### 8. KDE Plot
```python
sns.kdeplot(df['Study_Hours'], fill=True)
```

### 🧠 Key Concepts for Viva

| Concept | Explanation |
|---------|-------------|
| **Data Wrangling** | Process of cleaning, transforming, and restructuring raw data for analysis |
| **IQR (Interquartile Range)** | Q3 - Q1. Measures spread of middle 50% of data |
| **Outlier** | Data point beyond `Q1 - 1.5*IQR` or `Q3 + 1.5*IQR` |
| **Skewness** | Asymmetry of distribution. Right-skew = long right tail |
| **Log Transformation** | `log1p(x)` reduces right skewness by compressing large values |
| **Q-Q Plot** | Quantile-Quantile plot — if data is normal, points lie on a straight diagonal line |
| **KDE Plot** | Kernel Density Estimate — smooth curve showing probability distribution |
| **Median imputation** | Preferred over mean for skewed data as median is robust to outliers |

### ❓ Likely Viva Questions
1. **What is data wrangling?**  
   It is the process of cleaning, structuring, and enriching raw data into a usable format for analysis and ML models.
2. **Why use IQR for outlier detection?**  
   IQR is robust to extreme values and doesn't assume normality, unlike z-score method.
3. **Why is log transformation used?**  
   To reduce right skewness by compressing large values, making the distribution closer to normal — beneficial for many ML algorithms that assume normality.
4. **What is a Q-Q plot and when is it useful?**  
   A Q-Q plot compares data quantiles against theoretical normal quantiles. Points aligning on a straight line indicate normality.

---

## 🔷 DS03 — Descriptive Statistics on Adult Census & Iris Dataset

### 📌 Objective
Compute and interpret descriptive statistical measures (mean, median, std, percentiles) on two datasets: Adult Census Income and Iris.

### 📦 Datasets
1. **Adult Census Income** (`uciml/adult-census-income`) — 32,561 records, income classification
2. **Iris Dataset** (`uciml/iris`) — 150 records, 3 species of iris flowers

### 🔧 Part 1 — Adult Census Dataset

#### Key Statistical Operations
```python
# Summary statistics
df[num_cols].describe()
df[num_cols].mean()
df[num_cols].median()
df[num_cols].std()
df[num_cols].min()
df[num_cols].max()

# Grouped statistics
df.groupby("income")["age"].describe()
df.groupby("income")["age"].mean()
df.groupby("marital.status")["age"].mean()

# Value counts
df["marital.status"].value_counts()
df["income"].value_counts()
```

#### Numeric Columns Analyzed
| Column | Description |
|--------|-------------|
| age | Age of individual |
| hours.per.week | Work hours per week |
| capital.gain | Capital gain |
| capital.loss | Capital loss |

### 🔧 Part 2 — Iris Dataset

#### Key Operations
```python
# Per-species statistics
setosa = iris[iris['Species'] == 'Iris-setosa'].describe()

# Percentiles by group
iris.groupby("Species")["SepalLengthCm"].quantile(0.25)  # Q1
iris.groupby("Species")["SepalLengthCm"].quantile(0.50)  # Median
iris.groupby("Species")["SepalLengthCm"].quantile(0.75)  # Q3
```

#### Visualizations
```python
# Boxplot: Sepal/Petal length by Species
sns.boxplot(x='Species', y='SepalLengthCm', data=iris)

# Barplot: Mean Sepal Length by Species
sns.barplot(x='Species', y='SepalLengthCm', data=iris)
```

### 🧠 Key Concepts for Viva

| Concept | Explanation |
|---------|-------------|
| **Mean** | Average value. Sensitive to outliers |
| **Median** | Middle value when sorted. Robust to outliers |
| **Standard Deviation** | Measures spread of data around mean |
| **Percentile / Quantile** | Value below which X% of data falls |
| **`groupby()`** | Groups data by a categorical column for aggregate statistics |
| **`describe()`** | One-shot summary: count, mean, std, min, Q1, Q2, Q3, max |
| **Iris Dataset** | Benchmark dataset with 3 species (setosa, versicolor, virginica) and 4 features |

### ❓ Likely Viva Questions
1. **What is the difference between mean and median?**  
   Mean is the arithmetic average (affected by outliers). Median is the middle value (robust to outliers). Use median when data is skewed.
2. **What does `groupby()` do in pandas?**  
   It splits data into groups based on a column's values, applies an aggregation function (mean, sum, count), and combines results.
3. **What are the 3 species in the Iris dataset?**  
   Iris-setosa, Iris-versicolor, Iris-virginica.
4. **What is standard deviation?**  
   It measures how much individual values deviate from the mean. Low std = data is clustered; High std = data is spread out.

---

## 🔷 DS04 — Linear Regression on Boston Housing Dataset

### 📌 Objective
Build a Linear Regression model to predict house prices using the Boston Housing dataset, evaluate using RMSE and R² score, and visualize predictions and residuals.

### 📦 Dataset
**Boston Housing Dataset** (`vikrishnan/boston-house-prices`) — 506 records, 14 feature columns.

| Column | Description |
|--------|-------------|
| CRIM | Crime rate per town |
| RM | Average number of rooms per dwelling |
| LSTAT | % of lower-status population |
| PRICE | Median house price (target variable) |
| ... | 10 other socio-economic features |

### 🔧 Steps Performed

#### 1. Correlation Analysis
```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# RM has highest positive correlation with PRICE
# LSTAT has highest negative correlation with PRICE
```

#### 2. Prepare Features and Target
```python
X = df.drop('PRICE', axis=1)   # 13 features
Y = df['PRICE']                 # Target variable
```

#### 3. Train-Test Split
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=10)
# 80% training, 20% testing
```

#### 4. Train Model
```python
reg = LinearRegression()
reg.fit(X_train, Y_train)
# Learns: PRICE = b0 + b1*CRIM + b2*RM + ... + b13*LSTAT
```

#### 5. Predict and Evaluate
```python
Y_pred = reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = reg.score(X_test, Y_test)
```

#### 6. Visualization
```python
# Actual vs Predicted scatter plot
plt.scatter(Y_test, Y_pred)
plt.plot([min, max], [min, max], color='red')  # Perfect prediction line

# Residual plot
residuals = Y_test - Y_pred
sns.scatterplot(x=Y_pred, y=residuals)
plt.axhline(y=0, color='red')  # Zero error reference line
```

### 🧠 Key Concepts for Viva

| Concept | Explanation |
|---------|-------------|
| **Linear Regression** | Models relationship: `Y = b0 + b1*X1 + b2*X2 + ...` where b's are learned coefficients |
| **RMSE** | Root Mean Squared Error — average prediction error in same units as target |
| **R² Score** | Coefficient of determination. R²=1 means perfect fit; R²=0 means model is no better than mean |
| **Residual** | Difference between actual and predicted value: `Y_actual - Y_predicted` |
| **Correlation Heatmap** | Shows linear correlation between all feature pairs (-1 to +1) |
| **Train-Test Split** | Divides data; model trains on 80% and is evaluated on unseen 20% |
| **Overfitting** | Model learns training data too well but performs poorly on test data |
| **`reg.coef_`** | Array of learned coefficients for each feature |
| **`reg.intercept_`** | The bias term (b0) in the regression equation |

### ❓ Likely Viva Questions
1. **What is Linear Regression?**  
   A supervised ML algorithm that models the linear relationship between one or more independent variables (X) and a continuous dependent variable (Y).
2. **What does R² = 0.75 mean?**  
   The model explains 75% of the variance in house prices. Remaining 25% is unexplained by the features chosen.
3. **What is RMSE and why is it preferred over MSE?**  
   RMSE is the square root of Mean Squared Error, expressed in the same unit as the target variable — making it easier to interpret.
4. **What does the residual plot tell you?**  
   If residuals are randomly scattered around zero → model assumptions are satisfied. Patterns in residuals indicate a non-linear relationship that linear regression cannot capture.
5. **What does the red diagonal line in actual vs predicted plot represent?**  
   It represents perfect prediction (actual = predicted). Points closer to this line = better model.

---

## 🔷 DS05 — Logistic Regression on Social Network Ads Dataset

### 📌 Objective
Build a Logistic Regression model to classify whether a user purchased a product based on Age and Estimated Salary. Evaluate using a confusion matrix and compute classification metrics.

### 📦 Dataset
**Social Network Ads Dataset** — Features: Age, EstimatedSalary; Target: Purchased (0 or 1).

### 🔧 Steps Performed

#### 1. Load and Prepare Data
```python
X = dataset.iloc[:, [2, 3]].values   # Age, EstimatedSalary
y = dataset.iloc[:, 4].values        # Purchased (0/1)
```

#### 2. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# 75% train, 25% test
```

#### 3. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

#### 4. Train Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

#### 5. Confusion Matrix and Metrics
```python
cm = confusion_matrix(y_test, y_pred)
# [[TN, FP],
#  [FN, TP]]

accuracy  = (TP + TN) / (TP + TN + FP + FN)
error_rate = 1 - accuracy
precision  = TP / (TP + FP)
recall     = TP / (TP + FN)
```

#### 6. Decision Boundary Visualization
```python
# Meshgrid to plot decision regions
X1, X2 = np.meshgrid(np.arange(...), np.arange(...))
plt.contourf(X1, X2, model.predict(...).reshape(X1.shape), alpha=0.75)
plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set)
plt.title('Logistic Regression (Training set)')
```

### 🧠 Key Concepts for Viva

| Concept | Explanation |
|---------|-------------|
| **Logistic Regression** | Used for binary classification. Outputs probability using sigmoid function |
| **Sigmoid Function** | `σ(z) = 1 / (1 + e^(-z))`. Maps any value to (0,1) range |
| **Confusion Matrix** | 2x2 table: TP, FP, FN, TN showing classification results |
| **Accuracy** | `(TP+TN)/(TP+TN+FP+FN)` — overall correct predictions |
| **Precision** | `TP/(TP+FP)` — of all predicted positives, how many are actually positive |
| **Recall (Sensitivity)** | `TP/(TP+FN)` — of all actual positives, how many did model catch |
| **F1 Score** | Harmonic mean of precision and recall |
| **Standard Scaling** | `(x - mean) / std` — standardizes features to zero mean and unit variance |
| **Decision Boundary** | Line (or surface) that separates classes in feature space |

### ❓ Likely Viva Questions
1. **What is the difference between Linear and Logistic Regression?**  
   Linear regression predicts continuous values. Logistic regression predicts class probabilities (0 or 1) using the sigmoid function.
2. **What is a confusion matrix?**  
   A table comparing actual vs predicted class labels. It gives TP, FP, FN, TN — the basis for all classification metrics.
3. **When should you use precision vs recall?**  
   Use **precision** when false positives are costly (e.g., spam detection). Use **recall** when false negatives are costly (e.g., cancer diagnosis).
4. **Why do we apply feature scaling in logistic regression?**  
   Logistic regression uses gradient descent which converges faster with scaled features on the same range.
5. **What is TP, TN, FP, FN?**  
   - **TP** = Correctly predicted positive  
   - **TN** = Correctly predicted negative  
   - **FP** = Predicted positive but actually negative (Type I Error)  
   - **FN** = Predicted negative but actually positive (Type II Error)

---

## 🔷 DS06 — Naive Bayes Classifier on Iris Dataset

### 📌 Objective
Apply the Gaussian Naive Bayes classifier to the Iris dataset, evaluate accuracy, display confusion matrix, and compute performance metrics.

### 📦 Dataset
**Iris Dataset** — loaded from GitHub (Plotly datasets).  
3 classes: Iris-setosa, Iris-versicolor, Iris-virginica.  
4 features: sepal length, sepal width, petal length, petal width.

### 🔧 Steps Performed

#### 1. Load Dataset
```python
data = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/iris-data.csv")
```

#### 2. Prepare Features and Target
```python
X = data.drop(['class'], axis=1)     # All 4 features
y = data[['class']]                   # Species column
```

#### 3. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
```

#### 4. Train Gaussian Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

#### 5. Evaluate Model
```python
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

print(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
```

#### 6. Compute Metrics
```python
TP, FP, FN, TN = get_confusion_matrix_values(y_test, y_pred)
accuracy  = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
```

### 🧠 Key Concepts for Viva

| Concept | Explanation |
|---------|-------------|
| **Naive Bayes** | Probabilistic classifier based on Bayes' theorem with "naive" independence assumption between features |
| **Bayes' Theorem** | `P(A|B) = P(B|A) × P(A) / P(B)` |
| **Gaussian Naive Bayes** | Assumes each feature follows a Gaussian (normal) distribution within each class |
| **Prior Probability** | P(class) — probability of a class before seeing evidence |
| **Likelihood** | P(feature | class) — probability of feature given class |
| **Posterior Probability** | P(class | features) — what we compute to classify |
| **"Naive" Assumption** | Features are assumed to be independent of each other given the class |
| **Multiclass Classification** | More than 2 output classes (here: 3 species) |

### ❓ Likely Viva Questions
1. **What is Naive Bayes?**  
   A probabilistic classifier using Bayes' theorem. It calculates the probability of each class given the input features and selects the class with highest posterior probability.
2. **Why is it called "Naive"?**  
   Because it naively assumes all features are conditionally independent given the class — which is rarely true in practice, but the algorithm still works well.
3. **What is Gaussian Naive Bayes?**  
   A variant that assumes continuous features follow a Gaussian (normal/bell curve) distribution.
4. **What are the advantages of Naive Bayes?**  
   Fast, simple, works well with small datasets, handles multi-class naturally, performs well even with the independence assumption violated.
5. **What does `model.score()` return?**  
   It returns the mean accuracy of the model on the given test data and labels.

---

## 🔷 DS07 — NLP Text Preprocessing with NLTK

### 📌 Objective
Perform fundamental Natural Language Processing (NLP) operations: Tokenization, Stemming, Lemmatization, POS Tagging, and Stop Word Removal.

### 📦 Input Text
```
Real madrid is set to win the UCL for the season.
Benzema might win Balon dor. Salah might be the runner up.
```

### 🔧 Steps Performed

#### 1. Download NLTK Resources
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
```

#### 2. Tokenization — Split Text
```python
# Sentence tokenization
tokens_sents = nltk.sent_tokenize(text)
# Output: ['Real madrid is set...', 'Benzema might...', 'Salah might...']

# Word tokenization
tokens_words = nltk.word_tokenize(text)
# Output: ['Real', 'madrid', 'is', 'set', 'to', ...]
```

#### 3. Stemming — Reduce to Root Form
```python
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stem = [ps.stem(word) for word in tokens_words]
# 'winning' → 'win', 'running' → 'run'
```

#### 4. Lemmatization — Reduce to Dictionary Form
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
leme = [lemmatizer.lemmatize(word) for word in stem]
# More accurate than stemming: 'better' → 'good'
```

#### 5. POS Tagging — Part of Speech
```python
print(nltk.pos_tag(leme))
# Output: [('Real', 'NNP'), ('madrid', 'NN'), ('set', 'VBD'), ...]
# NNP = Proper Noun, VBD = Verb Past Tense, etc.
```

#### 6. Stop Word Removal
```python
from nltk.corpus import stopwords
sw_nltk = stopwords.words('english')
# ['i', 'me', 'my', 'myself', 'we', 'is', 'to', 'the', ...]

words = [word for word in text.split() if word.lower() not in sw_nltk]
new_text = " ".join(words)
# Removes 'is', 'the', 'to', 'for', 'might', 'be', 'up'
```

### 🧠 Key Concepts for Viva

| Concept | Explanation |
|---------|-------------|
| **Tokenization** | Breaking text into smaller units — sentences or words |
| **Stemming** | Crudely chopping word to its root (may not be a real word): "running" → "run" |
| **Lemmatization** | Reducing word to its dictionary base form using vocabulary: "ran" → "run" |
| **POS Tagging** | Assigning grammatical labels (Noun, Verb, Adjective, etc.) to each word |
| **Stop Words** | Common words with little meaning (is, the, a, to) that are removed before analysis |
| **Corpus** | A large collection of text data used for NLP training |
| **NLTK** | Natural Language Toolkit — Python library for NLP tasks |
| **`punkt`** | NLTK tokenizer model for splitting text into sentences/words |

#### POS Tag Reference
| Tag | Meaning |
|-----|---------|
| NN | Noun (singular) |
| NNP | Proper Noun |
| VB | Verb (base) |
| VBD | Verb (past tense) |
| JJ | Adjective |
| RB | Adverb |
| IN | Preposition |

### ❓ Likely Viva Questions
1. **What is NLP?**  
   Natural Language Processing — a branch of AI that enables computers to understand, interpret, and generate human language.
2. **What is the difference between stemming and lemmatization?**  
   Stemming is faster but crude (may produce non-words). Lemmatization is slower but more accurate — always produces a valid dictionary word.
3. **Why remove stop words?**  
   Stop words (is, the, a) are very frequent but carry little meaning, so removing them reduces noise and improves model efficiency.
4. **What is POS Tagging and why is it useful?**  
   POS tagging identifies the grammatical role of each word. It's used in machine translation, information extraction, and text parsing.
5. **What stemmers are available in NLTK?**  
   PorterStemmer, SnowballStemmer, LancasterStemmer — each with different aggressiveness levels.

---

## 🔷 DS08 — Data Visualization on Titanic Dataset

### 📌 Objective
Perform comprehensive exploratory data visualization on the Titanic dataset using various Seaborn and Matplotlib chart types.

### 📦 Dataset
Titanic dataset loaded from GitHub (dphi-official).

### 🔧 Steps Performed

#### 1. Data Cleaning (Pre-visualization)
```python
data['Age'] = data['Age'].fillna(np.mean(data['Age']))
data['Cabin'] = data['Cabin'].fillna(data['Cabin'].mode()[0])
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
```

#### 2. Count Plots (Categorical Data)
```python
sns.countplot(data['Survived'])    # class distribution
sns.countplot(data['Pclass'])      # passenger class distribution
sns.countplot(data['Embarked'])    # port distribution
sns.countplot(data['Sex'])         # gender distribution
```

#### 3. Box Plots (Distribution + Outliers)
```python
sns.boxplot(data['Age'])
sns.boxplot(data['Fare'])
sns.catplot(x='Pclass', y='Age', data=data, kind='box')
sns.catplot(x='Pclass', y='Fare', data=data, kind='strip')
```

#### 4. Pair Plot (All Feature Relationships)
```python
sns.pairplot(data)
```

#### 5. Scatter Plot (Two Variables)
```python
sns.scatterplot(x='Fare', y='Pclass', hue='Survived', data=data)
```

#### 6. Distribution Plot (Continuous Data)
```python
sns.distplot(data['Age'])
sns.distplot(data['Fare'])
```

#### 7. Joint Plot (Bivariate Analysis)
```python
sns.jointplot(x="Survived", y="Fare", kind="scatter", data=data)
```

#### 8. Heatmap (Correlation Matrix)
```python
tc = data.corr()
sns.heatmap(tc, cmap="YlGnBu")
plt.title('Correlation')
```

#### 9. Bar Plot (Grouped Aggregation)
```python
sns.catplot(x='Pclass', y='Fare', data=data, kind='bar')
```

### 🧠 Key Concepts for Viva

| Chart Type | When to Use |
|------------|-------------|
| **Count Plot** | Frequency of categorical variables |
| **Box Plot** | Distribution, median, IQR, outliers of continuous data |
| **Scatter Plot** | Relationship between two continuous variables |
| **Pair Plot** | All pairwise relationships between numerical columns |
| **Heatmap** | Correlation strength between all numerical features |
| **Distribution Plot** | Histogram + KDE for single continuous variable |
| **Joint Plot** | Bivariate distribution with marginal distributions |
| **Strip Plot** | Individual data points for categorical vs numerical |
| **Bar Plot** | Aggregated mean (or other) of continuous vs categorical |

### ❓ Likely Viva Questions
1. **What is a pair plot?**  
   It creates a grid of scatter plots for all pairs of numerical columns, with histograms on the diagonal — useful for quick EDA.
2. **What does a heatmap show?**  
   A color-coded matrix showing correlation coefficients between all numerical features. Dark colors = strong correlation.
3. **What is the difference between distplot and histplot?**  
   `distplot` (deprecated) shows histogram + KDE curve. `histplot` is the modern replacement with similar functionality.
4. **What does `hue='Survived'` do in a scatter plot?**  
   It color-codes the points by the 'Survived' column, allowing visual separation of survived vs not survived.
5. **What is a box plot and what does it show?**  
   It shows: median (orange line), IQR box (Q1 to Q3), whiskers (1.5×IQR), and outliers as individual points.

---

## 🔷 DS09 — Box Plot: Age Distribution by Gender and Survival (Titanic)

### 📌 Objective
Plot a grouped box plot showing the distribution of age with respect to gender, further subdivided by survival status — then write observations.

### 📦 Dataset
Titanic dataset (same as DS08) loaded from GitHub.

### 🔧 Steps Performed

#### 1. Load and Clean Data
```python
data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/titanic_data.csv')

data['Age'] = data['Age'].fillna(np.mean(data['Age']))
data['Cabin'] = data['Cabin'].fillna(data['Cabin'].mode()[0])
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
```

#### 2. EDA
```python
data.head()
data.describe()
data.info()
data.isnull().sum()
```

#### 3. Key Visualization
```python
sns.boxplot(
    x=data['Sex'],
    y=data['Age'],
    hue=data['Survived'],
    palette='Set2'
).set_title('Distribution of Age by Gender and Survival')
plt.show()
```

### 📊 Observations (Viva Answer)
1. **Female survivors** were generally younger across all age groups.
2. **Older males** had a lower survival rate — younger males also had lower survival.
3. **Children (both genders)** had higher survival rates — supporting "women and children first" policy.
4. The **median age of female survivors is lower** than non-survivors.
5. **Male non-survivors** had a wide age range — suggesting age alone wasn't the deciding factor for males.

### 🧠 Key Concepts for Viva

| Concept | Explanation |
|---------|-------------|
| **Grouped Box Plot** | Box plot with `x=category`, `y=continuous`, `hue=another_category` |
| **`hue` parameter** | Colors/groups boxes by a third variable within each x-axis category |
| **palette='Set2'** | A color palette for distinguishing hue groups |
| **Box Plot Elements** | Median line, IQR box, whiskers, outlier points |
| **"Women and Children First"** | Historical maritime evacuation protocol — visible in survival data |

### ❓ Likely Viva Questions
1. **What does `hue` mean in seaborn box plots?**  
   It adds a third dimension by splitting each category group by another variable, using different colors.
2. **What can you infer from this box plot about survival?**  
   Females had a much higher survival rate. Among females, younger women survived more. Males of all ages had lower survival overall.
3. **Why is box plot preferred for this type of analysis?**  
   Box plots show the entire distribution (median, spread, outliers) for each sub-group — cleaner than histograms for multi-group comparison.

---

## 🔷 DS10 — Feature Distribution Analysis on Iris Dataset

### 📌 Objective
Explore the Iris dataset using histograms with KDE curves and box plots to understand the distribution of each feature and compare across species.

### 📦 Dataset
Iris dataset loaded from GitHub (curran/iris.csv).  
Features: sepal_length, sepal_width, petal_length, petal_width, species.

### 🔧 Steps Performed

#### 1. Load and Explore
```python
data = pd.read_csv('https://gist.githubusercontent.com/curran/...iris.csv')
data.head()
data.describe()
data.describe(include='object')
data.isnull().sum()
```

#### 2. Print Feature Data Types
```python
print("1. Sepal length : ", data['sepal_length'].dtype)   # float64
print("2. Sepal width  : ", data['sepal_width'].dtype)
print("3. Petal length : ", data['petal_length'].dtype)
print("4. Petal width  : ", data['petal_width'].dtype)
print("5. Species      : ", data['species'].dtype)         # object
```

#### 3. Histogram + KDE for Each Feature
```python
sns.histplot(x=data['sepal_length'], kde=True)
sns.histplot(x=data['sepal_width'],  kde=True)
sns.histplot(x=data['petal_length'], kde=True)
sns.histplot(x=data['petal_width'],  kde=True)
```

#### 4. Box Plots for Outlier Detection
```python
sns.boxplot(data['sepal_length'])
sns.boxplot(data['sepal_width'])
sns.boxplot(data['petal_length'])
sns.boxplot(data['petal_width'])
```

#### 5. Box Plots Grouped by Species
```python
sns.boxplot(x='sepal_length', y='species', data=data)
sns.boxplot(x='petal_length', y='species', data=data)
```

### 📊 Observations
- **Petal length and petal width** show clear separation between species → most useful features for classification.
- **Sepal width** shows overlap between species → less discriminating.
- **Setosa** has distinctly smaller petal dimensions compared to versicolor and virginica.
- **Petal length histogram** is bimodal → indicates two distinct groups.

### 🧠 Key Concepts for Viva

| Concept | Explanation |
|---------|-------------|
| **Histogram** | Bars showing frequency count of data in value bins |
| **KDE Curve** | Smooth probability density curve overlaid on histogram |
| **Bimodal Distribution** | Distribution with two peaks — indicates two distinct subgroups |
| **Box Plot (Species-wise)** | Comparing feature distribution across all three iris species |
| **Feature Importance** | Features with clear separation between classes are more useful for classification |
| **`describe(include='object')`** | Summary statistics for categorical columns (count, unique, top, freq) |

### ❓ Likely Viva Questions
1. **What is a histogram?**  
   A bar chart where each bar represents the frequency of values falling within a value range (bin).
2. **What does `kde=True` do in `sns.histplot`?**  
   It overlays a Kernel Density Estimation curve — a smooth estimate of the probability distribution.
3. **What is a bimodal distribution and what does it indicate in the Iris dataset?**  
   A distribution with two peaks. In Iris, petal length has two peaks because setosa has very small petals, while versicolor/virginica have larger ones.
4. **Which Iris feature is best for classification and why?**  
   Petal length and petal width — they show the least overlap between species and the clearest separation in box plots.
5. **What does `data.describe(include='object')` return?**  
   It returns statistics for categorical columns: count, number of unique values, most frequent value (top), and its frequency (freq).

---

## 🛠️ Libraries Used (Quick Reference)

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, and analysis |
| `numpy` | Numerical computations, array operations |
| `matplotlib` | Base plotting library |
| `seaborn` | High-level statistical visualizations built on matplotlib |
| `sklearn` | Machine learning algorithms, preprocessing, metrics |
| `scipy` | Scientific computing — Q-Q plots, statistical tests |
| `nltk` | Natural Language Processing toolkit |
| `kagglehub` | Programmatically download Kaggle datasets |

---

## ⚡ Common Viva Questions Across All Practicals

| Question | Answer |
|----------|--------|
| What is `df.shape`? | Returns (rows, columns) tuple |
| What is `df.head(n)`? | Returns first n rows (default 5) |
| What is `df.isnull().sum()`? | Count of null values per column |
| What is `train_test_split()`? | Splits data into training and testing sets randomly |
| What is `fit()` vs `predict()`? | `fit()` trains the model; `predict()` applies it to new data |
| What is `fit_transform()` vs `transform()`? | `fit_transform()` learns parameters and transforms; `transform()` only transforms using learned parameters — use on test set |
| What is `random_state`? | Seed for reproducibility of random operations |
| What is overfitting? | Model memorizes training data — works well on train, poorly on test |
| What is underfitting? | Model is too simple — performs poorly on both train and test |
| What is cross-validation? | Technique to evaluate model by training/testing on different data splits |

---

*📝 Prepared by: Karan | TE — SEM 6 | D.Y. Patil College of Engineering*  
*📅 Academic Year: 2025–2026*
