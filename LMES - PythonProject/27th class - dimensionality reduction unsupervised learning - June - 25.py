import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = {
    'Name': ['Dhoni', 'Ashwin', 'Kohil', 'Raina'],
    'Maths': [90, 95, 85, 87],
    'English': [70, 93, 88, 78],
    'Tamil': [92, 85, 89, 90]
}

df = pd.DataFrame(data)

features = ['Maths', 'English', 'Tamil']
x = df.iloc[:, 1:4]
print(f'''First x output
      {x}''')

SS = StandardScaler()
x = SS.fit_transform(x)

print(f'''After fit and transformed output of x
{x} \n''')

PCA = PCA(n_components=2)
pca_model = PCA.fit_transform(x)

print(f''' PCA Model
{pca_model} \n''')

final_df = pd.DataFrame(pca_model, columns=['PCA_1', 'PCA_2'])

output = pd.concat([df[['Name']], final_df], axis=1)

print('Final output')
print(output)
print()
''' Finding the variance ratio '''
print(f'variance ratio', PCA.explained_variance_ratio_)
''' 
 variance ratio [0.74774337 0.24876739]
Which mean PCA_1 70% and PCA_2 30%
 '''
# Print PCA components to understand the contribution of each feature
print("PCA Components:")
print(PCA.components_)
print()

# Create a DataFrame to display feature contributions to each principal component
components_df = pd.DataFrame(PCA.components_.T, columns=['PCA_1', 'PCA_2'], index=features)
print("Feature contributions to each principal component:")
print(components_df)

components_df = pd.DataFrame(PCA.components_, columns=features, index=['PCA_1', 'PCA_2'])
print("\nFeature contributions to each principal component:")
print(components_df)


a = [1, 2, 3, 4, 1, 2, 3, 4]

print(a[2])

print(a.index(2, 3))

