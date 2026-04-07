import pandas as pd

id_genres = pd.read_csv('id_genres.csv', sep='\t')

id_genres['genre_list'] = id_genres['genres'].astype(str).str.split(',')#kano ta eidi lista kai xorizonatai me koma
                                                                        #p.x. 06pbFHF18RwKu4t1	jazz,blues,pop

top_5_genres = id_genres['genres'].value_counts().head(5).index.tolist()
print(top_5_genres)
