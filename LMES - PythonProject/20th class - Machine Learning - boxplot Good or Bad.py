import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

good_data = np.random.normal(loc=50, scale=5, size=100)
good_df = pd.DataFrame(good_data, columns=['Good_data_performance'])

bad_data = np.random.normal(loc=50, scale=20, size=100)
bad_data = np.append(bad_data, [150, 160, 170, 180, 190])

bad_df = pd.DataFrame(bad_data, columns=['Bad_Data_Performance'])

plt.figure(figure=(10, 8))

plt.subplot(1, 2, 1)
plt.boxplot(good_df['Good_data_performance'], vert=False)
plt.title('Testing cross validation - Good')

plt.subplot(1, 2, 2)
plt.boxplot(bad_df['Bad_Data_Performance'], vert=False)
plt.title('Testing cross validation - Bad')

plt.tight_layout()
plt.show()





























