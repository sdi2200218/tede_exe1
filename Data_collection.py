import pandas as pd
import tarfile
import os

print("Βήμα 1: Σύνδεση αρχείων...")
# Φόρτωση Genres
id_genres = pd.read_csv('id_genres.csv', sep='\t')
# Καθαρισμός διπλοτύπων (π.χ. pop,pop -> pop)
genres_cleaned = id_genres.assign(genres=id_genres['genres'].str.split(',')) \
    .explode('genres').assign(genres=lambda x: x['genres'].str.strip()).drop_duplicates()

# Εύρεση Top-5
top_5_genres = genres_cleaned['genres'].value_counts().head(5).index.tolist()
id_to_genre_map = genres_cleaned[genres_cleaned['genres'].isin(top_5_genres)].drop_duplicates(subset=['id'], keep='first')
target_ids = set(id_to_genre_map['id'])

# Φόρτωση Audio (Chunking)
audio_list = []
for chunk in pd.read_csv('id_mfcc_stats.tsv.bz2', sep='\t', compression='bz2', chunksize=2000):
    audio_list.append(chunk[chunk['id'].isin(target_ids)])
df_audio = pd.concat(audio_list)

# Φόρτωση Lyrics (Tar)
lyrics_data = []
with tarfile.open('processed_lyrics.tar.gz', "r:gz") as tar:
    for member in tar.getmembers():
        if member.isfile() and member.name.endswith('.txt'):
            song_id = os.path.basename(member.name).replace('.txt', '')
            if song_id in target_ids:
                f = tar.extractfile(member)
                if f:
                    lyrics_data.append({'id': song_id, 'lyrics': f.read().decode('utf-8', errors='ignore').strip()})
df_lyrics = pd.DataFrame(lyrics_data)

# Merge & Save
df_final = pd.merge(pd.merge(df_audio, df_lyrics, on='id'), id_to_genre_map[['id', 'genres']], on='id')
df_final = df_final.rename(columns={'genres': 'genre'})
df_final.to_csv('df_multimodal_sdi_26.csv', index=False)
print(f"Αρχικό dataset έτοιμο: {len(df_final)} τραγούδια.")