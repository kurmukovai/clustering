import pandas as pd
import numpy as np

def get_coauthors(df_data, per_year=True, year=None):
    '''
    input
    
    df_data: pandas Data Frame,
             with columns: title, year, author_id_new
             
    per_year: bool,
              if True, coauthors
              will contain coathors for each year,
              if False, see "year" parameter
    year: int, 
          if "per_year" is False return coauthors for given year,
          if None, return coauthors for all years
    
    return
    
    adjacency_matrix: numpy ndarray, 
                      adjacency coauthorship matrix
    authors: authors name list (indices are the same as in adjacency matrix)
               
    Currently ignore year
    '''
    
    title_author_counts = itas_data.groupby(by=['title', 'author_id_new']).count()
    
    titles, authors = title_author_counts.index.levels
    title_index, author_index = title_author_counts.index.labels
    
    incidence_matrix = pd.crosstab(author_index, title_index).values
    adjacency_matrix = incidence_matrix.dot(incidence_matrix.T)
    np.fill_diagonal(adjacency_matrix, 0)
    
    
    return adjacency_matrix, authors
