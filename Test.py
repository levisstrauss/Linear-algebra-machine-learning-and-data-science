import numpy as np

# Define matrices A and B
A = np.array([[1, 0],
              [2, 4]])

B = np.array([[1, 0],
              [0, 1]])

# Calculate eigenvalues and eigenvectors for A
eigenvalues_A, eigenvectors_A = np.linalg.eig(A)

# Calculate eigenvalues and eigenvectors for B
eigenvalues_B, eigenvectors_B = np.linalg.eig(B)

print(eigenvectors_A , " test" , eigenvectors_B)