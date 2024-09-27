import unittest
from main import *
import numpy as np
import pandas as pd
from scipy import sparse

class TestStringMethods(unittest.TestCase):

    def test_preprocessing(self):

        self.text_content= "Given that indexing is so important as your data set increases in size, can someone explain how indexing works at a database-agnostic level? For information on queries to index a field, check out How do I index a database column"
        self.text_result1='given that indexing is so important as your data set increases in size can someone explain how indexing works at a database agnostic level for information on queries to index a field check out how do i index a database column'
        
        self.assertEqual(preprocessing(self.text_content), self.text_result1)

    def test_lemmatize(self):
        self.text_content= "given that indexing is so important as your data set increases in size can someone explain how indexing works at a database agnostic level for information on queries to index a field check out how do i index a database column"
        self.text_result1= 'given that indexing is so important a your data set increase in size can someone explain how indexing work at a database agnostic level for information on query to index a field check out how do i index a database column'
 
        self.assertEqual(lemmatize(self.text_content), self.text_result1)

    def test_replace_characters(self):
        self.text_content= 'given that indexing is so important a your data set increase in size can someone explain how indexing work at a database agnostic level for information on query to index a field check out how do i index a database column :) <3'
        self.text_result2= 'given that indexing is so important a your data set increase in size can someone explain how indexing work at a database agnostic level for information on query to index a field check out how do i index a database column <3'

        self.assertEqual(replace_characters(self.text_content), self.text_result2)

    def test_vectorize(self):
        self.content = 'hello'
        X = vectorize(self.content, tfidf).toarray()
        X = pd.DataFrame(X.tolist())
        L = tfidf.get_feature_names_out().tolist()
        self.assertEqual(X.iloc[0,L.index(self.content)], 1.0)

    def test_predict_sentence(self):
        x_feat = tfidf.get_feature_names_out().tolist()
        x = [0.0] * 5000
        x[x_feat.index('python')]=1.0
        x = np.array(x)
        matrix = sparse.coo_matrix((x))
        self.assertEqual(predict_sentence(matrix, ovr_dt), [('python',)])

if __name__ == '__main__':
    unittest.main()