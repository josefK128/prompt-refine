# cosine_similarity.py
# calculates and returns the cosine similarity of two equal-length
# numpy ndarray vectors


import numpy as np

class CosineSimilarity:
    @staticmethod
    def action(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculates and returns the cosine similarity between two vectors.

        Args:
            vector1 (np.ndarray): First vector.
            vector2 (np.ndarray): Second vector.

        Returns:
            float: Cosine similarity between the two vectors.
        """
        if len(vector1) != len(vector2):
            raise ValueError("Vectors must have the same length.")
        
        #print(f'zero test on {vector1} is {np.all((vector1 == 0))}')
        #print(f'zero test on {vector2} is {np.all((vector2 == 0))}')
        if( np.all((vector1 == 0))): 
            raise ValueError("vector1 the zero-vector!")
        if( np.all((vector2 == 0))): 
            raise ValueError("vector2 the zero-vector!")

        dot_product = np.dot(vector1, vector2)
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)

        if norm_product == 0:
            raise ValueError("Vectors must not be zero vectors.")

        return dot_product / norm_product
