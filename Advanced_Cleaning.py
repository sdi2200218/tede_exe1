import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import RobustScaler

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    # Καθαρισμός στίχων: μόνο γράμματα, lowercase, αφαίρεση stopwords και lemmatization
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    stop_words = set(stopwords.words('english'))
    stop_words.update(['yeah', 'oh', 'ooh', 'la', 'baby', 'got', 'get', 'know'])
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2])

print("Έναρξη Preprocessing και Δειγματοληψίας...")
df = pd.read_csv('df_multimodal_sdi_26.csv')

# 1. NLP Καθαρισμός
df['lyrics'] = df['lyrics'].apply(clean_text)
df = df[df['lyrics'].str.len() > 20] # Φιλτράρισμα πολύ μικρών στίχων

# 2. Audio Scaling (Robust Scaler για προστασία από outliers)
mfcc_cols = [c for c in df.columns if c not in ['id', 'genre', 'lyrics']]
scaler = RobustScaler()
df[mfcc_cols] = scaler.fit_transform(df[mfcc_cols])

# 3. Ισορροπημένη Δειγματοληψία (2000 ανά είδος = 10.000 σύνολο)
df_balanced = df.groupby('genre', group_keys=False).apply(lambda x: x.sample(min(len(x), 2000), random_state=8312))
df_balanced = df_balanced.sample(frac=1, random_state=8312).reset_index(drop=True)

# 4. ΜΟΡΦΟΠΟΙΗΣΗ ΣΤΗΛΩΝ ΒΑΣΕΙ ΕΚΦΩΝΗΣΗΣ (id, genre, lyrics, mfcc_stats)
# Δημιουργούμε μια στήλη 'mfcc_stats' που περιέχει όλα τα audio features ως λίστα/string
# Αυτό διασφαλίζει ότι το CSV έχει ακριβώς τις 4 στήλες που ζητούνται.
print("Μορφοποίηση στηλών για το τελικό CSV...")
df_balanced['mfcc_stats'] = df_balanced[mfcc_cols].values.tolist()

# Κρατάμε μόνο τις 4 στήλες της οδηγίας
df_final_export = df_balanced[['id', 'genre', 'lyrics', 'mfcc_stats']]

# Αποθήκευση
df_final_export.to_csv('df_multimodal_sdi_26.csv', index=False)

print("-" * 30)
print(f"ΤΟ ΤΕΛΙΚΟ CSV ΕΙΝΑΙ ΕΤΟΙΜΟ!")
print(f"Αρχείο: df_multimodal_sdi_26.csv")
print(f"Στήλες: {df_final_export.columns.tolist()}")
print(f"Συνολικά τραγούδια: {len(df_final_export)}")