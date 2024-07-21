import pandas as pd
import matplotlib.pyplot as plt

data = {'Values1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'Values2': [100, 10, 600, 700, 60, 30, 50, 75, 90, 10]}

df = pd.DataFrame(data)


q1 = df['Values1'].quantile(0.25)
q2 = df['Values1'].median()
q3 = df['Values1'].quantile(0.74)

print(f''' IQR
           25% : {q1} \n
           50% : {q2} \n
           75% : {q3}
''')

'''*******************IQR*******************'''

IQR = q3 - q1

lower_fence = q1 = 1.5*IQR
higher_fence = q3 + 1.5*IQR

print(f'''
Lower Fence: {lower_fence}
Higher Fence: {higher_fence}
''')


fig, axis = plt.subplots(1, 2, figsize=(10, 5))
axis[0].boxplot(df['Values1'], vert=False)
axis[1].boxplot(df['Values2'], vert=False)

plt.tight_layout()
plt.show()




