from ast import alias
from concurrent.futures import process
from django.shortcuts import render, HttpResponse
from django.contrib import messages

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})



def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

import os
from django.conf import settings
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
def DatasetView(request):
    from django.conf import settings
    import pandas as pd 
    path = settings.MEDIA_ROOT + "//" + 'balanced_urgency_levels.csv'
    d = pd.read_csv(path)   
    # Drop the last column
    if not d.empty:
        d = d.iloc[:]  
    # d = d.head(50)  
    print(d)
    return render(request,'users/DatasetView.html', {'d': d})


#========================================================================================================
import os
import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE  

# Paths
MODEL_PATH_BEST = os.path.join(settings.MEDIA_ROOT, 'best_model.pkl')
TFIDF_PATH = os.path.join(settings.MEDIA_ROOT, 'tfidf_vectorizer.pkl')
ENCODERS_PATH = os.path.join(settings.MEDIA_ROOT, 'encoder_urgency.pkl')
DATASET_PATH = os.path.join(settings.MEDIA_ROOT, 'balanced_urgency_levels.csv')


def load_data():
    df = pd.read_csv(DATASET_PATH)
    df['Caller Statement'] = df['Caller Statement'].fillna('')
    
    le_urgency = LabelEncoder()
    df['Urgency Level'] = le_urgency.fit_transform(df['Urgency Level'])
    
    return df, le_urgency


def handle_imbalance(X, y):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X, y)


def train_model(request):
    df, le_urgency = load_data()
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['Caller Statement'])
    y = df['Urgency Level']
    X_resampled, y_resampled = handle_imbalance(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=39)
    
    # Train Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    # Train SVM
    svm_model = SVC(probability=True)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    
    # Train ANN
    ann_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(set(y_train)), activation='softmax')
    ])
    ann_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ann_model.fit(X_train.toarray(), y_train, epochs=10, batch_size=32, verbose=0)
    ann_preds = np.argmax(ann_model.predict(X_test.toarray()), axis=1)

    # Evaluate Models
    models = {
        "Random Forest": (rf_model, rf_preds),
        "SVM": (svm_model, svm_preds),
        "ANN": (ann_model, ann_preds)
    }
    
    metrics = {}
    best_model, best_acc = None, 0

    for name, (model, preds) in models.items():
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted')
        rec = recall_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')
        
        try:
            auc_roc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
        except:
            auc_roc = "N/A"

        metrics[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "AUC_ROC": auc_roc
        }
        
        if acc > best_acc:
            best_model, best_acc = model, acc

    joblib.dump(best_model, MODEL_PATH_BEST)
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump(le_urgency, ENCODERS_PATH)
    
    return render(request, 'users/train.html', {'metrics': metrics})



def predict_urgency(request):
    if request.method == 'POST':
        callerStatement = request.POST.get('callerStatement')
        best_model = joblib.load(MODEL_PATH_BEST)
        tfidf = joblib.load(TFIDF_PATH)
        le_urgency = joblib.load(ENCODERS_PATH)
        caller_statement_tfidf = tfidf.transform([callerStatement])
        
        if isinstance(best_model, Sequential):
            predicted_urgency = np.argmax(best_model.predict(caller_statement_tfidf.toarray()), axis=1)
        else:
            predicted_urgency = best_model.predict(caller_statement_tfidf)
        
        predicted_label = le_urgency.inverse_transform(predicted_urgency)[0]
        return render(request, 'users/ML.html', {'predicted_urgency': predicted_label})
    
    return render(request, 'users/ML.html', {'error': 'Invalid request method'})