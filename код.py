import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm

df = pd.read_csv('flights_NY.csv').dropna()


# 1
df['Положительная задержка'] = df['arr_delay'].apply(lambda x: 1 if x > 0 else 0)
dc = df.groupby('carrier')['Положительная задержка'].mean().sort_values(ascending=True)

plt.figure(figsize=(10, 6))
dc.plot(kind='bar')

plt.title('Распределение вероятности положительной задержки по авиакомпаниям')
plt.xlabel('Авиакомпания')
plt.ylabel('Вероятность положительной задержки')
plt.tight_layout()

plt.show()


# 2
print("№2")

plt.figure(figsize=(10, 6))
plt.hist(df['distance'], bins=50, color='skyblue', edgecolor='black')
plt.title('Распределение расстояния перелета')
plt.xlabel('Расстояние, миль')
plt.ylabel('Частота')
plt.grid(True)
plt.show()

quantiles = df['distance'].quantile([0.25, 0.5, 0.75])
print("Квантили для определения границ групп:\n", quantiles)

short = quantiles.loc[0.25]
medium = quantiles.loc[0.5]
long = quantiles.loc[0.75]

print("\nГраницы групп:")
print("Короткие перелеты: до", short, "миль")
print("Средние перелеты: от", short, "до", medium, "миль")
print("Длинные перелеты: от", medium, "миль и выше")

category = ['Короткий', 'Средний', 'Длинный']

df['Категория перелета'] = pd.cut(df['distance'], bins=[0, quantiles[0.25], quantiles[0.5], df['distance'].max()])

lf = df[df['Категория перелета'] == 'Длинный']['dest'].unique()
print("Направления для длинных перелетов:", lf)

ad = df.groupby('Категория перелета', observed=False)['dep_delay'].mean()
print("Среднее время задержки вылета по категориям перелетов в минутах:\n", ad)


# 3
df['month'] = pd.to_datetime(df['month'], format='%m').dt.month_name()

plt.figure(figsize=(10, 6))
sns.pointplot(data=df, x='month', y='dep_delay', errorbar=('ci', 95))
plt.title('Среднее время задержки вылета по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Среднее время задержки вылета (мин)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

january = df[df['month'] == 'January']['dep_delay']
february = df[df['month'] == 'February']['dep_delay']

t_statistic, p_value = stats.ttest_ind(january, february)

print("\n№3")
alpha = 0.05
if p_value < alpha:
    print("уровень значимости 0.05: отвергается")
else:
    print("уровень значимости 0.05: не отвергается")

alpha = 0.01
if p_value < alpha:
    print("уровень значимости 0.01: отвергается")
else:
    print("уровень значимости 0.01: не отвергается")


# 4
print("\n№4")

corr = df['distance'].corr(df['air_time'])
print("Коэффициент корреляции между расстоянием и временем полета:", corr)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='distance', y='air_time')
plt.title('Точечная диаграмма распределения расстояния и времени полета')
plt.xlabel('Расстояние, миль')
plt.ylabel('Время полета, мин')
plt.grid(True)

slope, intercept, r, p, std = stats.linregress(df['distance'], df['air_time'])

x = np.array([df['distance'].min(), df['distance'].max()])
y = slope * x + intercept
plt.plot(x, y, color='red', linewidth=2)

plt.show()

print("Коэффициенты линейной регрессии:")
print("slope:", slope)
print("intercept:", intercept)


# 5
print("\n№5")

dfn = df[(df['dep_delay'] >= -15) & (df['dep_delay'] <= 15)]

plt.figure(figsize=(10, 6))
sns.histplot(dfn['arr_delay'], bins=30, kde=True, stat='density')
plt.title('Нормированная гистограмма распределения задержки прилета')
plt.xlabel('Задержка прилета, мин')
plt.ylabel('Плотность вероятности')

mu, stdd = norm.fit(dfn['arr_delay'])

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
pr = norm.pdf(x, mu, stdd)
plt.plot(x, pr, 'k', linewidth=2)
plt.legend(['Нормальное распределение ($\\mu$={:.2f}, $\\sigma$={:.2f})'.format(mu, stdd), 'Гистограмма'])
plt.grid(True)
plt.show()

print("Оцененные параметры нормального распределения:")
print("Среднее (mu):", mu)
print("Стандартное отклонение (sigma):", stdd)