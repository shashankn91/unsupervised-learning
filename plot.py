import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

out = './{}/'.format(sys.argv[1])

df = pd.read_csv(out + 'digits2D.csv')
Z = pd.DataFrame(df['target'])

reduced_data = df.drop(['target'], axis = 1)
h = .02
x_min, x_max = df['x'].min() - 1, df['x'].max() + 1
y_min, y_max = df['y'].min() - 1, df['y'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
centers = df.groupby('target')['x', 'y'].mean()



colors = ['b','g','r','c','m','y','k','w']

def func(x):
    return colors[int(x%len(colors))]
pppp = df['target'].apply(func )


plt.scatter(df['x'], df['y'], c=pppp,s=50, cmap='viridis')

plt.scatter(centers['x'], centers['y'], c='black', s=200, alpha=0.5)
plt.savefig(out + 'digits.png')


#Madelon  Plot
df = pd.read_csv(out + 'madelon2D.csv')
Z = pd.DataFrame(df['target'])

reduced_data = df.drop(['target'], axis = 1)
h = .02
x_min, x_max = df['x'].min() - 1, df['x'].max() + 1
y_min, y_max = df['y'].min() - 1, df['y'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
centers = df.groupby('target')['x', 'y'].mean()


pppp = df['target'].apply(func )


plt.scatter(df['x'], df['y'], c=pppp,s=50, cmap='viridis')

plt.scatter(centers['x'], centers['y'], c='black', s=200, alpha=0.5)
plt.savefig(out + 'madelon.png')