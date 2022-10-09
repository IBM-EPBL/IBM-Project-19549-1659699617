{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7c07e186",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn import preprocessing\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "import plotly.express as px\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "import wget"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "065f1675",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100% [............................................................................] 684858 / 684858"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'Churn_Modelling.csv'"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wget.download(\"https://drive.google.com/u/0/uc?id=1_HcM0K8wt4b7FMLkc1V1dv0y6I_9ULzy&export=download\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3cd9c1c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('C:/Users/Dell/PycharmProjects/Android-Malware-Prediction-in-Machine-Learning/Churn_Modelling.csv', index_col=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "07a159ff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>Surname</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>RowNumber</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>15634602</td>\n",
       "      <td>Hargrave</td>\n",
       "      <td>619</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101348.88</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15647311</td>\n",
       "      <td>Hill</td>\n",
       "      <td>608</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>112542.58</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>15619304</td>\n",
       "      <td>Onio</td>\n",
       "      <td>502</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113931.57</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>15701354</td>\n",
       "      <td>Boni</td>\n",
       "      <td>699</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>93826.63</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>15737888</td>\n",
       "      <td>Mitchell</td>\n",
       "      <td>850</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>43</td>\n",
       "      <td>2</td>\n",
       "      <td>125510.82</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>79084.10</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           CustomerId   Surname  CreditScore Geography  Gender  Age  Tenure  \\\n",
       "RowNumber                                                                     \n",
       "1            15634602  Hargrave          619    France  Female   42       2   \n",
       "2            15647311      Hill          608     Spain  Female   41       1   \n",
       "3            15619304      Onio          502    France  Female   42       8   \n",
       "4            15701354      Boni          699    France  Female   39       1   \n",
       "5            15737888  Mitchell          850     Spain  Female   43       2   \n",
       "\n",
       "             Balance  NumOfProducts  HasCrCard  IsActiveMember  \\\n",
       "RowNumber                                                        \n",
       "1               0.00              1          1               1   \n",
       "2           83807.86              1          0               1   \n",
       "3          159660.80              3          1               0   \n",
       "4               0.00              2          0               0   \n",
       "5          125510.82              1          1               1   \n",
       "\n",
       "           EstimatedSalary  Exited  \n",
       "RowNumber                           \n",
       "1                101348.88       1  \n",
       "2                112542.58       0  \n",
       "3                113931.57       1  \n",
       "4                 93826.63       0  \n",
       "5                 79084.10       0  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "877d1735",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',\n",
       "       'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',\n",
       "       'EstimatedSalary', 'Exited'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "3db8fae2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 13)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "455cd610",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1], dtype=int64)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.unique(df['Exited'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "a25ea0bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "#5. Handling Missing values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "cd026c9d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CustomerId         0\n",
       "Surname            0\n",
       "CreditScore        0\n",
       "Geography          0\n",
       "Gender             0\n",
       "Age                0\n",
       "Tenure             0\n",
       "Balance            0\n",
       "NumOfProducts      0\n",
       "HasCrCard          0\n",
       "IsActiveMember     0\n",
       "EstimatedSalary    0\n",
       "Exited             0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a5f556d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "#4. Perform descriptive statistics on the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "ebf9499b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>Surname</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1.000000e+04</td>\n",
       "      <td>10000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000</td>\n",
       "      <td>10000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.00000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>NaN</td>\n",
       "      <td>2932</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>NaN</td>\n",
       "      <td>Smith</td>\n",
       "      <td>NaN</td>\n",
       "      <td>France</td>\n",
       "      <td>Male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>NaN</td>\n",
       "      <td>32</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5014</td>\n",
       "      <td>5457</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1.569094e+07</td>\n",
       "      <td>NaN</td>\n",
       "      <td>650.528800</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>38.921800</td>\n",
       "      <td>5.012800</td>\n",
       "      <td>76485.889288</td>\n",
       "      <td>1.530200</td>\n",
       "      <td>0.70550</td>\n",
       "      <td>0.515100</td>\n",
       "      <td>100090.239881</td>\n",
       "      <td>0.203700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>7.193619e+04</td>\n",
       "      <td>NaN</td>\n",
       "      <td>96.653299</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>10.487806</td>\n",
       "      <td>2.892174</td>\n",
       "      <td>62397.405202</td>\n",
       "      <td>0.581654</td>\n",
       "      <td>0.45584</td>\n",
       "      <td>0.499797</td>\n",
       "      <td>57510.492818</td>\n",
       "      <td>0.402769</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.556570e+07</td>\n",
       "      <td>NaN</td>\n",
       "      <td>350.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>11.580000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>1.562853e+07</td>\n",
       "      <td>NaN</td>\n",
       "      <td>584.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>51002.110000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1.569074e+07</td>\n",
       "      <td>NaN</td>\n",
       "      <td>652.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>37.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>97198.540000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>100193.915000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.575323e+07</td>\n",
       "      <td>NaN</td>\n",
       "      <td>718.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>44.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>127644.240000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>149388.247500</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.581569e+07</td>\n",
       "      <td>NaN</td>\n",
       "      <td>850.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>92.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>250898.090000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>199992.480000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          CustomerId Surname   CreditScore Geography Gender           Age  \\\n",
       "count   1.000000e+04   10000  10000.000000     10000  10000  10000.000000   \n",
       "unique           NaN    2932           NaN         3      2           NaN   \n",
       "top              NaN   Smith           NaN    France   Male           NaN   \n",
       "freq             NaN      32           NaN      5014   5457           NaN   \n",
       "mean    1.569094e+07     NaN    650.528800       NaN    NaN     38.921800   \n",
       "std     7.193619e+04     NaN     96.653299       NaN    NaN     10.487806   \n",
       "min     1.556570e+07     NaN    350.000000       NaN    NaN     18.000000   \n",
       "25%     1.562853e+07     NaN    584.000000       NaN    NaN     32.000000   \n",
       "50%     1.569074e+07     NaN    652.000000       NaN    NaN     37.000000   \n",
       "75%     1.575323e+07     NaN    718.000000       NaN    NaN     44.000000   \n",
       "max     1.581569e+07     NaN    850.000000       NaN    NaN     92.000000   \n",
       "\n",
       "              Tenure        Balance  NumOfProducts    HasCrCard  \\\n",
       "count   10000.000000   10000.000000   10000.000000  10000.00000   \n",
       "unique           NaN            NaN            NaN          NaN   \n",
       "top              NaN            NaN            NaN          NaN   \n",
       "freq             NaN            NaN            NaN          NaN   \n",
       "mean        5.012800   76485.889288       1.530200      0.70550   \n",
       "std         2.892174   62397.405202       0.581654      0.45584   \n",
       "min         0.000000       0.000000       1.000000      0.00000   \n",
       "25%         3.000000       0.000000       1.000000      0.00000   \n",
       "50%         5.000000   97198.540000       1.000000      1.00000   \n",
       "75%         7.000000  127644.240000       2.000000      1.00000   \n",
       "max        10.000000  250898.090000       4.000000      1.00000   \n",
       "\n",
       "        IsActiveMember  EstimatedSalary        Exited  \n",
       "count     10000.000000     10000.000000  10000.000000  \n",
       "unique             NaN              NaN           NaN  \n",
       "top                NaN              NaN           NaN  \n",
       "freq               NaN              NaN           NaN  \n",
       "mean          0.515100    100090.239881      0.203700  \n",
       "std           0.499797     57510.492818      0.402769  \n",
       "min           0.000000        11.580000      0.000000  \n",
       "25%           0.000000     51002.110000      0.000000  \n",
       "50%           1.000000    100193.915000      0.000000  \n",
       "75%           1.000000    149388.247500      0.000000  \n",
       "max           1.000000    199992.480000      1.000000  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe(include='all')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "669eb340",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#3. Perform different Visualizations.\n",
    "\n",
    "    # Univariate Analysis\n",
    "    # Bi - Variate Analysis\n",
    "    # Multi - Variate Analysis\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "3ffd13ed",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x26c42b6cac0>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiMAAAGdCAYAAADAAnMpAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy89olMNAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAjG0lEQVR4nO3de3RU1cH38V8uZJI8kASNmXAZDCoKCAImEAPaLpejKbKw9EqREkoVC0ULpI9KBBJbK6FeKFZQKlWxT0UQl1IraXhpBC0aiQSiIDctYrLQSaCUTAxIILPfP5TRkQQzYTKbhO9nrVnqmX3O2bMTMl8nM4cIY4wRAACAJZG2JwAAAM5txAgAALCKGAEAAFYRIwAAwCpiBAAAWEWMAAAAq4gRAABgFTECAACsirY9gZbw+Xz6+OOP1aVLF0VERNieDgAAaAFjjOrq6tS9e3dFRjb/+ke7iJGPP/5YLpfL9jQAAEArVFVVqWfPns3e3y5ipEuXLpI+fzAJCQmWZwMAAFrC6/XK5XL5n8eb0y5i5OSvZhISEogRAADamW96iwVvYAUAAFYRIwAAwCpiBAAAWEWMAAAAq4gRAABgFTECAACsIkYAAIBVxAgAALCqXVz0rC0cbWjUvX/fpnXbPtahz0xQ+0ZICm6PpkVK8gWxvbVaM9/m9omQlBQboQip2XUL1fqEQ7SkRjU/35hIKS5aOtIgHT+D80R+cY5QrUuUvpjXidN/r0Tp8/tP/l9HY4jO31JRkjpFSsd9LT/3V79/Tu5/zNeytYuWFBUpNXwx/uvfi5FfHPPrX8uIL/Y9obPzezcyQoqLjtBFF/yPfCdO6L2az1q0X4Sk+Gjp+AmpoQXjO+nMvs+/Libi8zkca+WiBvuzJEJSlxjJKEJ1DXa+kv8TLZ0XH60q74lmx3z1cUVK6txJOnq89Wsf/cU6x8dEKsL4dPhrX+zT/TyP0Od/ZjrHdlL25akqGH254mKiWjmT1gv6lZHXX39do0ePVvfu3RUREaHVq1d/4z4bNmzQlVdeKYfDoUsuuUTLli1rxVRDZ/Jf3la//GKtfHt/0CEihe6HVXNPIqEMEal1821uHyPpv5+Z067b2fjDvDnf9OTT4JNqzzBEpM+/pqFcl0ZJn35DiJwcZ774Z7hD5OT5PwsiRKTAdTq5f0vX7oQCw+Xr+/nU9NfSfLH9bP3e9Rmp/rjRto8/bXGISJ8/nvoWhogU2hCRpAbT+hCRgv96GEneBlkLEenz9T5diEiBj8snyXsGISJJJ4x03Ei1x04Nka+f7+vbffr8fxb+e+S4VrxdpX75xZr8l7fPYDatE3SM1NfXa9CgQVq8eHGLxn/44YcaNWqUrr32WlVUVGjGjBm69dZbtXbt2qAnGwqT//K21u2osXJuAADOdut21IQ9SIL+Nc3IkSM1cuTIFo9fsmSJevfurYcffliS1K9fP23cuFF/+MMflJ2dHezpz8jRhkZCBACAb7BuR42ONjSG7Vc2bf4G1tLSUrnd7oBt2dnZKi0tbXafY8eOyev1BtxCYV7RjpAcBwCAji6cz5ltHiMej0dOpzNgm9PplNfr1dGjR5vcp7CwUImJif6by+UKyVz2/edISI4DAEBHF87nzLPyo715eXmqra3136qqqkJy3LTz40NyHAAAOrpwPme2eYykpqaquro6YFt1dbUSEhIUFxfX5D4Oh0MJCQkBt1C458b+ITkOAAAdXTifM9s8RrKyslRSUhKwbd26dcrKymrrU58iLiZK1/dPCft5AQBoT67vnxLW640EHSOffvqpKioqVFFRIenzj+5WVFSosrJS0ue/YsnJyfGPnzJlivbu3au77rpLu3bt0mOPPabnn39eM2fODM0jCNLSnKEECQAAzbi+f4qW5gwN6zmD/mjv5s2bde211/r/Ozc3V5I0ceJELVu2TJ988ok/TCSpd+/eWrNmjWbOnKlHHnlEPXv21J///Oewf6z3q5bmDOUKrK3chyuwtg5XYOUKrGeCK7C2fDxXYG2fV2CNMMacjX/2Ani9XiUmJqq2tjZk7x8BAABtq6XP32flp2kAAMC5gxgBAABWESMAAMAqYgQAAFhFjAAAAKuIEQAAYBUxAgAArCJGAACAVcQIAACwihgBAABWESMAAMAqYgQAAFhFjAAAAKuIEQAAYBUxAgAArCJGAACAVcQIAACwihgBAABWESMAAMAqYgQAAFhFjAAAAKuIEQAAYBUxAgAArCJGAACAVcQIAACwihgBAABWESMAAMAqYgQAAFhFjAAAAKuIEQAAYBUxAgAArCJGAACAVcQIAACwihgBAABWESMAAMAqYgQAAFhFjAAAAKuIEQAAYBUxAgAArCJGAACAVcQIAACwihgBAABWESMAAMAqYgQAAFhFjAAA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/XDOOuXl5aqpqdGVV16p6OhoRUdH67XXXtMf//hHRUdHy+l0ss4h0K1bN/Xv3z9gW79+/VRZWSnpy3U63c+I1NRU1dTUBNx/4sQJHTp0KKivRUd25513+l8dGThwoCZMmKCZM2f6X/VjndtGONe1uTHBrvs5GSMxMTFKT09XSUmJf5vP51NJSYmysrIszuzsZYzR7bffrpdeekmvvvqqevfuHXB/enq6OnXqFLCmu3fvVmVlpX9Ns7KytG3btoA/AOvWrVNCQoL/iSErKyvgGCfHnCtfl+uuu07btm1TRUWF/5aRkaHx48f7/511PnMjRow45aPpe/bs0YUXXihJ6t27t1JTUwPWyOv1atOmTQHrfPjwYZWXl/vHvPrqq/L5fMrMzPSPef3113X8+HH/mHXr1umyyy5T165d2+zxnS2OHDmiyMjAp5moqCj5fD5JrHNbCee6huxnSVBvd+1AVqxYYRwOh1m2bJnZsWOHue2220xSUlLAJxDwpalTp5rExESzYcMG88knn/hvR44c8Y+ZMmWK6dWrl3n11VfN5s2bTVZWlsnKyvLff/IjpzfccIOpqKgwxcXF5oILLmjyI6d33nmn2blzp1m8ePE59ZHTpnz10zTGsM6hUFZWZqKjo839999v3n//ffPss8+a+Ph489e//tU/Zv78+SYpKcn87W9/M++++6757ne/2+RHI4cMGWI2bdpkNm7caPr06RPw0cjDhw8bp9NpJkyYYLZv325WrFhh4uPjO/RHTr9q4sSJpkePHv6P9r744osmOTnZ3HXXXf4xrHPr1NXVma1bt5qtW7caSWbBggVm69at5qOPPjLGhG9d33jjDRMdHW0eeughs3PnTlNQUMBHe4P16KOPml69epmYmBgzbNgw89Zbb9me0llLUpO3p59+2j/m6NGj5pe//KXp2rWriY+PN9/73vfMJ598EnCcffv2mZEjR5q4uDiTnJxsfv3rX5vjx48HjFm/fr0ZPHiwiYmJMRdddFHAOc5FX48R1jk0/v73v5sBAwYYh8Nh+vbta5544omA+30+n5k7d65xOp3G4XCY6667zuzevTtgzH/+8x8zbtw407lzZ5OQkGAmTZpk6urqAsa888475uqrrzYOh8P06NHDzJ8/v80f29nC6/Wa6dOnm169epnY2Fhz0UUXmdmzZwd8VJR1bp3169c3+TN54sSJxpjwruvzzz9vLr30UhMTE2Muv/xys2bNmqAfT4QxX7kUHgAAQJidk+8ZAQAAZw9iBAAAWEWMAAAAq4gRAABgFTECAACsIkYAAIBVxAgAALCKGAEAAFYRIwAAwCpiBAAAWEWMAAAAq4gRAABg1f8H1vDJgn9L7tsAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(df.index,df['Exited'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "32c9d23e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x26c42d6dcd0>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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pPqZM2daklhYWFhYWLy2VFRUNDvHWsUz66bKzMzUlClT3OsNDQ06fPiwunbt2vz/K/r/XC6XoqOj9fXXX8tut7e01UsO83N2zNGZMT9nxxyd2fman86dOzf72FYR1qGhoWrbtq1KSko8tpeUlJzyJQ6bzSabzeaxLTg42Ks92e12/kjOgPk5O+bozJifs2OOzsyk+WkVt8j09/dXYmKicnJy3NsaGhqUk5Mjh8Phw84AADi7VvHMWpKmTJmi9PR0XX311br22mv1/PPP69ixY+5PhwMAYKpWE9Z33XWXDh48qBkzZsjpdGrgwIFas2ZNiz5p2Bw2m00zZ8486WV2fIf5OTvm6MyYn7Njjs7MxPnxsyzL8nUTAADg9FrFe9YAAFzMCGsAAAxHWAMAYDjCGgAAwxHWF9CCBQt02WWXqUOHDkpKStKnn37q65ZabNasWfLz8/NY+vT5/uYBJ06cUEZGhrp27apOnTpp5MiRJ305TVFRkdLS0hQYGKiwsDBNnTr1pFtn5ubmatCgQbLZbOrVq5eysrJO6sWU+f3www916623KioqSn5+fnr77bc99luWpRkzZigyMlIBAQFKTk7Wl19+6VFz+PBhjR49Wna7XcHBwRo7dqz7e+Ubbdu2TTfccIM6dOig6OhozZs376Re3njjDfXp00cdOnRQ//79tXr16ib34m1nm5/777//pN+pYcOGedRcyvMzd+5cXXPNNercubPCwsJ0++23q7Cw0KPGpL+rc+nF285ljoYMGXLS79H48eM9ai6qOWr2F5WiSZYvX275+/tbr7zyirVz505r3LhxVnBwsFVSUuLr1lpk5syZVr9+/azi4mL3cvDgQff+8ePHW9HR0VZOTo61ZcsW67rrrrN++tOfuvfX1dVZV155pZWcnGxt3brVWr16tRUaGmplZma6a7766isrMDDQmjJlirVr1y7rhRdesNq2bWutWbPGXWPS/K5evdp67LHHrDfffNOSZL311lse+5966ikrKCjIevvtt60vvvjC+sUvfmHFxcVZVVVV7pphw4ZZAwYMsDZt2mR99NFHVq9evay7777bvb+iosIKDw+3Ro8ebe3YscP6xz/+YQUEBFj/8z//46755JNPrLZt21rz5s2zdu3aZU2fPt1q3769tX379ib1cqHnJz093Ro2bJjH79Thw4c9ai7l+UlNTbVe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\n",
      "text/plain": [
       "<Figure size 500x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.displot(df['Balance'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "07f36b21",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: xlabel='CreditScore'>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.rugplot(df['CreditScore'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "a3733e21",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: xlabel='CreditScore', ylabel='Balance'>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot(x=df['CreditScore'], y=df['Balance'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "ad6b0735",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: xlabel='IsActiveMember', ylabel='Exited'>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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zp49q6NHj+rs2bM6d+6cvv3tbwfs09jYqLFjxwasJScnt/DVAnA5xA1wlQsODtZbb72l3bt3680339QvfvEL/eQnP9GePXs0ZMgQFRQUqL6+XikpKf59LMuSz+fTwYMHNWzYMIWFhV3xHCEhIbr77rtVUFCge++9VwUFBUpPT1evXp/9N3X27FklJSXplVdeabbvwIEDm6098MADSklJ0QcffKAHHnig2eNnz56VJK1ZsybgvUmffU2+qHfv3v4/f34B9JfXfD5fa9+q/7W3bNmiwYMHBzzmdDoD7l9zzTWtfl4An+GaGwByOBy65ZZbtHTpUu3bt08hISHatGmTJGndunX693//d5WXl/tv+/fv12233ab8/HxJnx3ZKCoquuTzh4SEqKmp6SvnmDlzprZv364///nP+sMf/qCZM2f6H7vxxhv117/+VdHR0br++usDbi39OPkNN9ygG264QR988IFmzJjR7HGXy6VBgwbp8OHDzZ5vyJAhXznrV9m/f7/Onz/vv//HP/5RERERio2N1ahRo+R0OlVZWdnstb98PRKAtuPIDXCV27Nnj4qKijRp0iRFR0drz549qqmp0ciRI1VeXq6ysjK98sorGjFiRMB+06dP11NPPaWf/vSnysrKUnx8vH74wx9q3rx5CgkJ0Y4dO3TPPfdowIABiouL0549e1RRUaGIiAj179+/xVm+9a1vKSYmRjNnztSQIUMCjqjMnDlTzzzzjO666y499dRTuu666/S3v/1NGzdu1OOPP67rrruu2fP94Q9/0IULF9S3b98WX2/p0qX60Y9+pKioKN1xxx1qaGjQe++9p1OnTsnj8bT/i6rPTjHNmTNHixcvVkVFhXJycjR//nwFBQWpT58+euyxx7Rw4UL5fD7deuutqq2t1a5duxQZGamMjIwrem3gaseRG+AqFxkZqZ07d2rKlCkaNmyYFi9erGeffVaTJ0/WunXrNGrUqGZhI0nf/e53dfz4cW3dulXDhg3Tm2++qf3792vcuHFKTU3Vf/3Xf/lPKT322GMKDg7WqFGjNHDgQFVWVrY4i8Ph0PTp07V///6AozaSFB4erp07d+rrX/+6vve972nkyJGaM2eO6uvrFRkZ2eLzXXPNNZcMG0l68MEHtXbtWq1fv17x8fEaP368NmzY0CFHbiZOnKhvfOMb+ta3vqX09HR95zvf0ZNPPul/fNmyZVqyZIlyc3M1cuRI3XHHHdqyZUuHvDZwtXNY1hd+3hMAAKCH48gNAAAwCnEDAACMQtwAAACjEDcAAMAoxA0AADAKcQMAAIxC3AAAAKMQNwAAwCjEDQAAMApxAwAAjELcAAAAo/w/Oj09W+AHoawAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x='IsActiveMember', y='Exited', data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "1180e11e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 558.875x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.lmplot(x='Tenure', y='CreditScore', data=df, hue=\"Exited\", fit_reg=False);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "9f2b8759",
   "metadata": {},
   "outputs": [],
   "source": [
    "#7. Check for Categorical columns and perform encoding."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "1cad8949",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Female', 'Male'], dtype=object)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Gender'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "dc68ec6f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['France', 'Spain', 'Germany'], dtype=object)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Geography'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "d7d569c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Geography']=preprocessing.LabelEncoder().fit_transform(df['Geography'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "1b4e03ca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>Surname</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>RowNumber</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>15634602</td>\n",
       "      <td>Hargrave</td>\n",
       "      <td>619</td>\n",
       "      <td>0</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101348.88</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15647311</td>\n",
       "      <td>Hill</td>\n",
       "      <td>608</td>\n",
       "      <td>2</td>\n",
       "      <td>Female</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>112542.58</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>15619304</td>\n",
       "      <td>Onio</td>\n",
       "      <td>502</td>\n",
       "      <td>0</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113931.57</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>15701354</td>\n",
       "      <td>Boni</td>\n",
       "      <td>699</td>\n",
       "      <td>0</td>\n",
       "      <td>Female</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>93826.63</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>15737888</td>\n",
       "      <td>Mitchell</td>\n",
       "      <td>850</td>\n",
       "      <td>2</td>\n",
       "      <td>Female</td>\n",
       "      <td>43</td>\n",
       "      <td>2</td>\n",
       "      <td>125510.82</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>79084.10</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9996</th>\n",
       "      <td>15606229</td>\n",
       "      <td>Obijiaku</td>\n",
       "      <td>771</td>\n",
       "      <td>0</td>\n",
       "      <td>Male</td>\n",
       "      <td>39</td>\n",
       "      <td>5</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>96270.64</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9997</th>\n",
       "      <td>15569892</td>\n",
       "      <td>Johnstone</td>\n",
       "      <td>516</td>\n",
       "      <td>0</td>\n",
       "      <td>Male</td>\n",
       "      <td>35</td>\n",
       "      <td>10</td>\n",
       "      <td>57369.61</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101699.77</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9998</th>\n",
       "      <td>15584532</td>\n",
       "      <td>Liu</td>\n",
       "      <td>709</td>\n",
       "      <td>0</td>\n",
       "      <td>Female</td>\n",
       "      <td>36</td>\n",
       "      <td>7</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>42085.58</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9999</th>\n",
       "      <td>15682355</td>\n",
       "      <td>Sabbatini</td>\n",
       "      <td>772</td>\n",
       "      <td>1</td>\n",
       "      <td>Male</td>\n",
       "      <td>42</td>\n",
       "      <td>3</td>\n",
       "      <td>75075.31</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>92888.52</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10000</th>\n",
       "      <td>15628319</td>\n",
       "      <td>Walker</td>\n",
       "      <td>792</td>\n",
       "      <td>0</td>\n",
       "      <td>Female</td>\n",
       "      <td>28</td>\n",
       "      <td>4</td>\n",
       "      <td>130142.79</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>38190.78</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10000 rows × 13 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "           CustomerId    Surname  CreditScore  Geography  Gender  Age  Tenure  \\\n",
       "RowNumber                                                                       \n",
       "1            15634602   Hargrave          619          0  Female   42       2   \n",
       "2            15647311       Hill          608          2  Female   41       1   \n",
       "3            15619304       Onio          502          0  Female   42       8   \n",
       "4            15701354       Boni          699          0  Female   39       1   \n",
       "5            15737888   Mitchell          850          2  Female   43       2   \n",
       "...               ...        ...          ...        ...     ...  ...     ...   \n",
       "9996         15606229   Obijiaku          771          0    Male   39       5   \n",
       "9997         15569892  Johnstone          516          0    Male   35      10   \n",
       "9998         15584532        Liu          709          0  Female   36       7   \n",
       "9999         15682355  Sabbatini          772          1    Male   42       3   \n",
       "10000        15628319     Walker          792          0  Female   28       4   \n",
       "\n",
       "             Balance  NumOfProducts  HasCrCard  IsActiveMember  \\\n",
       "RowNumber                                                        \n",
       "1               0.00              1          1               1   \n",
       "2           83807.86              1          0               1   \n",
       "3          159660.80              3          1               0   \n",
       "4               0.00              2          0               0   \n",
       "5          125510.82              1          1               1   \n",
       "...              ...            ...        ...             ...   \n",
       "9996            0.00              2          1               0   \n",
       "9997        57369.61              1          1               1   \n",
       "9998            0.00              1          0               1   \n",
       "9999        75075.31              2          1               0   \n",
       "10000      130142.79              1          1               0   \n",
       "\n",
       "           EstimatedSalary  Exited  \n",
       "RowNumber                           \n",
       "1                101348.88       1  \n",
       "2                112542.58       0  \n",
       "3                113931.57       1  \n",
       "4                 93826.63       0  \n",
       "5                 79084.10       0  \n",
       "...                    ...     ...  \n",
       "9996              96270.64       0  \n",
       "9997             101699.77       0  \n",
       "9998              42085.58       1  \n",
       "9999              92888.52       1  \n",
       "10000             38190.78       0  \n",
       "\n",
       "[10000 rows x 13 columns]"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "fce78b5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Gender']=df['Gender'].map({'Male':0,'Female':1})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "4a938977",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>Surname</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>RowNumber</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>15634602</td>\n",
       "      <td>Hargrave</td>\n",
       "      <td>619</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101348.88</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15647311</td>\n",
       "      <td>Hill</td>\n",
       "      <td>608</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>112542.58</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>15619304</td>\n",
       "      <td>Onio</td>\n",
       "      <td>502</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>42</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113931.57</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>15701354</td>\n",
       "      <td>Boni</td>\n",
       "      <td>699</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>93826.63</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>15737888</td>\n",
       "      <td>Mitchell</td>\n",
       "      <td>850</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>43</td>\n",
       "      <td>2</td>\n",
       "      <td>125510.82</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>79084.10</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9996</th>\n",
       "      <td>15606229</td>\n",
       "      <td>Obijiaku</td>\n",
       "      <td>771</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>39</td>\n",
       "      <td>5</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>96270.64</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9997</th>\n",
       "      <td>15569892</td>\n",
       "      <td>Johnstone</td>\n",
       "      <td>516</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>35</td>\n",
       "      <td>10</td>\n",
       "      <td>57369.61</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101699.77</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9998</th>\n",
       "      <td>15584532</td>\n",
       "      <td>Liu</td>\n",
       "      <td>709</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>36</td>\n",
       "      <td>7</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>42085.58</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9999</th>\n",
       "      <td>15682355</td>\n",
       "      <td>Sabbatini</td>\n",
       "      <td>772</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>42</td>\n",
       "      <td>3</td>\n",
       "      <td>75075.31</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>92888.52</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10000</th>\n",
       "      <td>15628319</td>\n",
       "      <td>Walker</td>\n",
       "      <td>792</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>28</td>\n",
       "      <td>4</td>\n",
       "      <td>130142.79</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>38190.78</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10000 rows × 13 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "           CustomerId    Surname  CreditScore  Geography  Gender  Age  Tenure  \\\n",
       "RowNumber                                                                       \n",
       "1            15634602   Hargrave          619          0       1   42       2   \n",
       "2            15647311       Hill          608          2       1   41       1   \n",
       "3            15619304       Onio          502          0       1   42       8   \n",
       "4            15701354       Boni          699          0       1   39       1   \n",
       "5            15737888   Mitchell          850          2       1   43       2   \n",
       "...               ...        ...          ...        ...     ...  ...     ...   \n",
       "9996         15606229   Obijiaku          771          0       0   39       5   \n",
       "9997         15569892  Johnstone          516          0       0   35      10   \n",
       "9998         15584532        Liu          709          0       1   36       7   \n",
       "9999         15682355  Sabbatini          772          1       0   42       3   \n",
       "10000        15628319     Walker          792          0       1   28       4   \n",
       "\n",
       "             Balance  NumOfProducts  HasCrCard  IsActiveMember  \\\n",
       "RowNumber                                                        \n",
       "1               0.00              1          1               1   \n",
       "2           83807.86              1          0               1   \n",
       "3          159660.80              3          1               0   \n",
       "4               0.00              2          0               0   \n",
       "5          125510.82              1          1               1   \n",
       "...              ...            ...        ...             ...   \n",
       "9996            0.00              2          1               0   \n",
       "9997        57369.61              1          1               1   \n",
       "9998            0.00              1          0               1   \n",
       "9999        75075.31              2          1               0   \n",
       "10000      130142.79              1          1               0   \n",
       "\n",
       "           EstimatedSalary  Exited  \n",
       "RowNumber                           \n",
       "1                101348.88       1  \n",
       "2                112542.58       0  \n",
       "3                113931.57       1  \n",
       "4                 93826.63       0  \n",
       "5                 79084.10       0  \n",
       "...                    ...     ...  \n",
       "9996              96270.64       0  \n",
       "9997             101699.77       0  \n",
       "9998              42085.58       1  \n",
       "9999              92888.52       1  \n",
       "10000             38190.78       0  \n",
       "\n",
       "[10000 rows x 13 columns]"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "17133592",
   "metadata": {},
   "outputs": [],
   "source": [
    "#6. Find the outliers and replace the outliers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "ad8509cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count     10000.000000\n",
       "mean      76485.889288\n",
       "std       62397.405202\n",
       "min           0.000000\n",
       "25%           0.000000\n",
       "50%       97198.540000\n",
       "75%      127644.240000\n",
       "max      250898.090000\n",
       "Name: Balance, dtype: float64"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()['Balance']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "631e0122",
   "metadata": {},
   "outputs": [],
   "source": [
    "def impute_outliers_IQR(df):\n",
    "    qu1 = df.quantile(0.25)\n",
    "    qu3 = df.quantile(0.75)\n",
    "    iqr = qu3 - qu1\n",
    "\n",
    "    upper = df[~(df > (qu3 + 1.5 * iqr))].max()\n",
    "    lower = df[~(df < (qu1 - 1.5 * iqr))].min()\n",
    "\n",
    "    df = np.where(\n",
    "       df > upper, \n",
    "       df.mean(), \n",
    "       np.where(df < lower, df.mean(), df)\n",
    "    )\n",
    "\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "65669c7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Balance'] = impute_outliers_IQR(df['Balance'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "2e7499c2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>Surname</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>RowNumber</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>15634602</td>\n",
       "      <td>Hargrave</td>\n",
       "      <td>619</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101348.88</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15647311</td>\n",
       "      <td>Hill</td>\n",
       "      <td>608</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>112542.58</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>15619304</td>\n",
       "      <td>Onio</td>\n",
       "      <td>502</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>42</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113931.57</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>15701354</td>\n",
       "      <td>Boni</td>\n",
       "      <td>699</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>93826.63</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>15737888</td>\n",
       "      <td>Mitchell</td>\n",
       "      <td>850</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>43</td>\n",
       "      <td>2</td>\n",
       "      <td>125510.82</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>79084.10</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9996</th>\n",
       "      <td>15606229</td>\n",
       "      <td>Obijiaku</td>\n",
       "      <td>771</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>39</td>\n",
       "      <td>5</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>96270.64</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9997</th>\n",
       "      <td>15569892</td>\n",
       "      <td>Johnstone</td>\n",
       "      <td>516</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>35</td>\n",
       "      <td>10</td>\n",
       "      <td>57369.61</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101699.77</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9998</th>\n",
       "      <td>15584532</td>\n",
       "      <td>Liu</td>\n",
       "      <td>709</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>36</td>\n",
       "      <td>7</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>42085.58</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9999</th>\n",
       "      <td>15682355</td>\n",
       "      <td>Sabbatini</td>\n",
       "      <td>772</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>42</td>\n",
       "      <td>3</td>\n",
       "      <td>75075.31</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>92888.52</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10000</th>\n",
       "      <td>15628319</td>\n",
       "      <td>Walker</td>\n",
       "      <td>792</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>28</td>\n",
       "      <td>4</td>\n",
       "      <td>130142.79</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>38190.78</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10000 rows × 13 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "           CustomerId    Surname  CreditScore  Geography  Gender  Age  Tenure  \\\n",
       "RowNumber                                                                       \n",
       "1            15634602   Hargrave          619          0       1   42       2   \n",
       "2            15647311       Hill          608          2       1   41       1   \n",
       "3            15619304       Onio          502          0       1   42       8   \n",
       "4            15701354       Boni          699          0       1   39       1   \n",
       "5            15737888   Mitchell          850          2       1   43       2   \n",
       "...               ...        ...          ...        ...     ...  ...     ...   \n",
       "9996         15606229   Obijiaku          771          0       0   39       5   \n",
       "9997         15569892  Johnstone          516          0       0   35      10   \n",
       "9998         15584532        Liu          709          0       1   36       7   \n",
       "9999         15682355  Sabbatini          772          1       0   42       3   \n",
       "10000        15628319     Walker          792          0       1   28       4   \n",
       "\n",
       "             Balance  NumOfProducts  HasCrCard  IsActiveMember  \\\n",
       "RowNumber                                                        \n",
       "1               0.00              1          1               1   \n",
       "2           83807.86              1          0               1   \n",
       "3          159660.80              3          1               0   \n",
       "4               0.00              2          0               0   \n",
       "5          125510.82              1          1               1   \n",
       "...              ...            ...        ...             ...   \n",
       "9996            0.00              2          1               0   \n",
       "9997        57369.61              1          1               1   \n",
       "9998            0.00              1          0               1   \n",
       "9999        75075.31              2          1               0   \n",
       "10000      130142.79              1          1               0   \n",
       "\n",
       "           EstimatedSalary  Exited  \n",
       "RowNumber                           \n",
       "1                101348.88       1  \n",
       "2                112542.58       0  \n",
       "3                113931.57       1  \n",
       "4                 93826.63       0  \n",
       "5                 79084.10       0  \n",
       "...                    ...     ...  \n",
       "9996              96270.64       0  \n",
       "9997             101699.77       0  \n",
       "9998              42085.58       1  \n",
       "9999              92888.52       1  \n",
       "10000             38190.78       0  \n",
       "\n",
       "[10000 rows x 13 columns]"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "12d3dc60",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count     10000.000000\n",
       "mean      76485.889288\n",
       "std       62397.405202\n",
       "min           0.000000\n",
       "25%           0.000000\n",
       "50%       97198.540000\n",
       "75%      127644.240000\n",
       "max      250898.090000\n",
       "Name: Balance, dtype: float64"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()['Balance']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "04875563",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count     10000.000000\n",
       "mean     100090.239881\n",
       "std       57510.492818\n",
       "min          11.580000\n",
       "25%       51002.110000\n",
       "50%      100193.915000\n",
       "75%      149388.247500\n",
       "max      199992.480000\n",
       "Name: EstimatedSalary, dtype: float64"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()['EstimatedSalary']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "1d5c85eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['EstimatedSalary'] = impute_outliers_IQR(df['EstimatedSalary'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "8fdcac45",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count     10000.000000\n",
       "mean     100090.239881\n",
       "std       57510.492818\n",
       "min          11.580000\n",
       "25%       51002.110000\n",
       "50%      100193.915000\n",
       "75%      149388.247500\n",
       "max      199992.480000\n",
       "Name: EstimatedSalary, dtype: float64"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()['EstimatedSalary']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "5c7e4c32",
   "metadata": {},
   "outputs": [],
   "source": [
    "def find_outliers_IQR(df):\n",
    "    qu1=df.quantile(0.25)\n",
    "    qu3=df.quantile(0.75)\n",
    "    iqr = qu3 - qu1\n",
    "\n",
    "    outliers = df[((df < (qu1 - 1.5 * iqr)) | (df > (qu3 + 1.5 * iqr)))]\n",
    "    return outliers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "d54691ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "outliers = find_outliers_IQR(df['EstimatedSalary'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "9840bafc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Series([], Name: EstimatedSalary, dtype: float64)\n"
     ]
    }
   ],
   "source": [
    "print(outliers)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "47ee39f0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n",
      "nan\n",
      "nan\n"
     ]
    }
   ],
   "source": [
    "print(len(outliers))\n",
    "print(outliers.max())\n",
    "print(outliers.min())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "95cf318d",
   "metadata": {},
   "outputs": [],
   "source": [
    "outliers = find_outliers_IQR(df['Balance'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "dd570700",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n",
      "nan\n",
      "nan\n"
     ]
    }
   ],
   "source": [
    "print(len(outliers))\n",
    "print(outliers.max())\n",
    "print(outliers.min())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "33018180",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: ylabel='Balance'>"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(y=df['Balance'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "60a95a12",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: ylabel='EstimatedSalary'>"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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AoQqAAAACxCqAAAALECoAgAAsAChCgAAwAKEKgAAAAsQqgAAACxAqAIAALAAoQoAAMAChCoAAAALEKoAAAAsQKgCAACwAKEKAADAAoQqAAAACxCqAAAALECoAgAAsAChCgAAwAKEKgAAAAsQqgAAACxAqAIAALAAoQoAAMAChCoAAAALEKoAAAAsQKgCAACwgF9D1ebNm3XHHXcoNjZWNptNa9as8Rq32Wx13ubOnWvWtG3b9pzxZ5991us4O3bs0K233iqn06m4uDjNmTPnnF5Wr16tTp06yel0qmvXrnr77be9xg3D0NSpUxUTE6OQkBAlJSVp79691r0ZAAAgoPk1VJWVlal79+5avHhxneOHDh3yui1dulQ2m02DBw/2qps5c6ZX3aOPPmqOeTweDRgwQG3atFFBQYHmzp2r6dOna8mSJWZNfn6+hg4dqtGjR+uTTz5RWlqa0tLStGvXLrNmzpw5WrhwobKzs7VlyxY1b95cycnJqqiosPhdAQAAgaiJP588JSVFKSkp5x2Pjo722n7zzTfVr18/tWvXzmt/aGjoObVnLF++XFVVVVq6dKnsdruuvfZaFRYWat68eRozZowkacGCBRo4cKAmTJggSXr66aeVm5urRYsWKTs7W4ZhaP78+ZoyZYruvPNOSdJrr72mqKgorVmzRkOGDPH5PQAAAJeHgJlTVVJSonXr1mn06NHnjD377LNq1aqVevbsqblz5+rUqVPmmMvlUp8+fWS32819ycnJKioq0rFjx8yapKQkr2MmJyfL5XJJkoqLi+V2u71qwsPDlZCQYNbUpbKyUh6Px+sGAAAuT349U3UxXn31VYWGhuquu+7y2v/YY4+pV69eatmypfLz85WVlaVDhw5p3rx5kiS32634+Hivx0RFRZljERERcrvd5r6za9xut1l39uPqqqnL7NmzNWPGDB9eLQAACDQBE6qWLl2q4cOHy+l0eu3PzMw073fr1k12u10PP/ywZs+eLYfD8VO36SUrK8urP4/Ho7i4OD92BAAALpWAuPz3P//zPyoqKtKDDz54wdqEhASdOnVK+/fvl3R6XlZJSYlXzZntM/Owzldz9vjZj6urpi4Oh0NhYWFeNwAAcHkKiFD18ssvq3fv3urevfsFawsLCxUUFKTIyEhJUmJiojZv3qzq6mqzJjc3Vx07dlRERIRZk5eX53Wc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8vJyxcXF6bbbbtM999yjq6++WidPntTMmTOt7hEAAKDB8/m3/77/o8XS6Z8wuZjf/gMAALhc+BSqDMOoc82J7du3q2XLlj+6KQAAgEBzUXOqIiIiZLPZZLPZ9POf/9wrWNXU1OjEiRMaO3as5U0CAAA0dBcVqubPny/DMPTAAw9oxowZCg8PN8fsdrvatm3LYpgAAKBRuqhQNXLkSEmnVxi/6aab1LQpX6cHAACQfFxSoW/fvub9iooKVVVVeY0H6hpMAAAAvvJponp5ebnGjRunyMhINW/eXBEREV43AACAxsanUDVhwgRt2LBBL774ohwOh1566SXNmDFDsbGxeu2116zuEQAAoMHz6fLfP//5T7322mv6xS9+oVGjRunWW29Vhw4d1KZNGy1fvlzDhw+3uk8AAIAGzaczVUePHlW7du0knZ4/dfToUUnSLbfcos2bN1vXHQAAQIDwKVS1a9dOxcXFkqROnTpp1apVkk6fwWrRooVlzQEAAAQKn0LVqFGjtH37dknS5MmTtXjxYjmdTmVkZGjChAmWNggAABAIfJpTlZGRYd5PSkrSnj17VFBQoA4dOqhbt26WNQcAABAofApV39emTRu1adPGikMBAAAEJJ9D1datW/X+++/r8OHDqq2t9RqbN2/ej24MAAAgkPgUqp555hlNmTJFHTt2VFRUlNcPK599HwAAoLHwKVQtWLBAS5cu1f33329xOwAAAIHJp2//BQUF6eabb7a6FwAAgIDlU6jKyMjQ4sWLre4FAAAgYPl0+e/JJ59Uamqq2rdvr86dO6tp06Ze46+//rolzQEAAAQKn0LVY489pvfff1/9+vVTq1atmJwOAAAaPZ9C1auvvqr//u//VmpqqtX9AAAABCSf5lS1bNlS7du3t7oXAACAgOVTqJo+fbqmTZum8vJyq/sBAAAISD5d/lu4cKH27dunqKgotW3b9pyJ6tu2bbOkOQAAgEDhU6hKS0uzuA0AAIDA5lOomjZtmtV9AAAABDSf5lQBAADAW73PVLVs2VL/+7//q9atWysiIuIH16Y6evSoJc0BAAAEinqfqXr++ecVGhpq3v+hm5Xatm0rm812zi09PV2S9Itf/OKcsbFjx3od48CBA0pNTVWzZs0UGRmpCRMm6NSpU141GzduVK9eveRwONShQwctW7bsnF4WL16stm3byul0KiEhQR999JGlrxUAAASuep+pGjlypHn//vvvvxS91Gnr1q2qqakxt3ft2qVf/epX+s1vfmPue+ihhzRz5kxzu1mzZub9mpoapaamKjo6Wvn5+Tp06JBGjBihpk2b6plnnpEkFRcXKzU1VWPHjtXy5cuVl5enBx98UDExMUpOTpYkrVy5UpmZmcrOzlZCQoLmz5+v5ORkFRUVKTIy8lK/DQAAoIHzaU5VcHCwDh8+fM7+b7/9VsHBwT+6qbNdccUVio6ONm9r165V+/bt1bdvX7OmWbNmXjVhYWHm2LvvvqtPP/1Uf/3rX9WjRw+lpKTo6aef1uLFi1VVVSVJys7OVnx8vJ577jldc801GjdunO6++26vs27z5s3TQw89pFGjRqlz587Kzs5Ws2bNtHTpUktfLwAACEw+hSrDMOrcX1lZKbvd/qMa+iFVVVX661//qgceeMBrTtfy5cvVunVrdenSRVlZWV6LkrpcLnXt2lVRUVHmvuTkZHk8Hu3evdusSUpK8nqu5ORkuVwu83kLCgq8aoKCgpSUlGTW1KWyslIej8frBgAALk8XtaTCwoULJUk2m00vvfSSfvazn5ljNTU12rx5szp16mRth2dZs2aNjh8/7nX5cdiwYWrTpo1iY2O1Y8cOTZo0SUVFRXr99dclSW632ytQSTK33W73D9Z4PB6dPHlSx44dU01NTZ01e/bsOW+/s2fP1owZM3x+vQAAIHBcVKg6cznMMAxlZ2d7Xeqz2+1q27atsrOzre3wLC+//LJSUlIUGxtr7hszZox5v2vXroqJiVH//v21b98+v/8+YVZWljIzM81tj8ejuLg4P3YEAAAulYsKVcXFxZKkfv366fXXX1dERMQlaaouX375pd577z3zDNT5JCQkSJI+//xztW/fXtHR0ed8S6+kpESSFB0dbf7nmX1n14SFhSkkJETBwcEKDg6us+bMMericDjkcDjq9wIBAEBA82lO1fvvv+8VqGpqalRYWKhjx45Z1tj3vfLKK4qMjFRqauoP1hUWFkqSYmJiJEmJiYnauXOn18T63NxchYWFqXPnzmZNXl6e13Fyc3OVmJgo6fRZuN69e3vV1NbWKi8vz6wBAACNm0+havz48Xr55ZclnQ5Uffr0Ua9evRQXF6eNGzda2Z+k0wHmlVde0ciRI9Wkyf+dXNu3b5+efvppFRQUaP/+/Xrrrbc0YsQI9enTR926dZMkDRgwQJ07d9Z9992n7du365133tGUKVOUnp5unkUaO3asvvjiC02cOFF79uzRCy+8oFWrVikjI8N8rszMTP35z3/Wq6++qs8++0yPPPKIysrKNGrUKMtfLwAACDw+/fbf6tWr9dvf/laS9M9//lP79+/Xnj179Je//EW///3v9cEHH1ja5HvvvacDBw7ogQce8Npvt9v13nvvaf78+SorK1NcXJwGDx6sKVOmmDXBwcFau3atHnnkESUmJqp58+YaOXKk17pW8fHxWrdunTIyMrRgwQJdeeWVeumll8w1qiTp3nvv1ZEjRzR16lS53W716NFD69evP2fyOgAAaJxsxvnWR/gBTqdTn3/+ua688kqNGTNGzZo10/z581VcXKzu3buzdMB5eDwehYeHq7S01GstrUBy8uRJpaSkSJK+63WfFNzUzx0BABqkmmqFbvuLJCknJ0chISF+bsh39f389unyX1RUlD799FPV1NRo/fr1+tWvfiVJKi8vt3zxTwAAgEDg0+W/UaNG6Z577lFMTIxsNpu5KOaWLVsu6TpVAAAADZVPoWr69Onq0qWLvvrqK/3mN78xJ3wHBwcrKyvL0gYBAAACwUVd/rvttttUWloqSbr77rtVWVnptar67bffrsmTJ1vbIQAAQAC4qFD1zjvvqLKy0tx+5plndPToUXP71KlTKioqsq47AACAAHFRoer7XxT04YuDAAAAlyWfvv0HAAAAbxcVqmw2m2w22zn7AAAAGruL+vafYRi6//77zW/7VVRUaOzYsWrevLkkec23AgAAaEwuKlSNHDnSa/vMT9WcbcSIET+uIwAAgAB0UaHqlVdeuVR9AAAABDQmqgMAAFiAUAUAAGABQhUAAIAFCFUAAAAWIFQBAABYgFAFAABgAUIVAACABQhVAAAAFiBUAQAAWIBQBQAAYAFCFQAAgAUIVQAAABYgVAEAAFiAUAUAAGABQhUAAIAFCFUAAAAWIFQBAABYgFAFAABgAUIVAACABQhVAAAAFmjQoWr69Omy2Wxet06dOpnjFRUVSk9PV6tWrfSzn/1MgwcPVklJidcxDhw4oNTUVDVr1kyRkZGaMGGCTp065VWzceNG9erVSw6HQx06dNCyZcvO6WXx4sVq27atnE6nEhIS9NFHH12S1wwAAAJTE383cCHXXnut3nvv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qaCgwKsmKChISUlJZk1dKisr5fF4vG4AAODy1MTfDdRXbW2txo8fr5tvvlldunQx9w8bNkxt2rRRbGysduzYoUmTJqmoqEivv/66JMntdnsFKknmttvt/sEaj8ejkydP6tixY6qpqamzZs+ePeftefbs2ZoxY4bvLxoAAASMgAlV6enp2rVrl/71r3957R8zZox5v2vXroqJiVH//v21b98+tW/f/qdu00tWVpYyMzPNbY/Ho7i4OD92BAAALpWACFXjxo3T2rVrtXnzZl155ZU/WJuQkCBJ+vzzz9W+fXtFR0ef8y29kpISSVJ0dLT5n2f2nV0TFhamkJAQBQcHKzg4uM6aM8eoi8PhkMPhqN+LBAAAAa1Bz6kyDEPjxo3TG2+8oQ0bNig+Pv6CjyksLJQkxcTESJISExO1c+dOr2/p5ebmKiwsTJ07dzZr8vLyvI6Tm5urxMRESZLdblfv3r29ampra5WXl2fWAACAxq1Bn6lKT0/XihUr9Oabbyo0NNScAxUeHq6QkBDt27dPK1as0G233aZWrVppx44dysjIUJ8+fdStWzdJ0oABA9S5c2fdd999mjNnjtxut6ZMmaL09HTzLNLYsWO1aNEiTZw4UQ888IA2bNigVatWad26dWYvmZmZGjlypK677jrdcMMNmj9/vsrKyjRq1Kif/o0BAAANj9GASarz9sorrxiGYRgHDhww+vTpY7Rs2dJwOBxGhw4djAkTJhilpaVex9m/f7+RkpJihISEGK1btzaeeOIJo7q62qvm/fffN3r06GHY7XajXbt25nOc7U9/+pNx1VVXGXa73bjhhhuMDz/88KJeT2lpqSHpnP4ABLby8nKjb9++Rt++fY3y8nJ/twPAYvX9/A6odaoCHetUAZcn1qkCLm+X5TpVAAAADRWhCgAAwAKEKgAAAAsQqgAAACxAqAIAALAAoQoAAMAChCoAAAALEKoAAAAsQKgCAACwAKEKAADAAoQqAAAACxCqAAAALECoAgAAsAChCgAAwAKEKgAAAAsQqgAAACxAqAIAALAAoQoAAMAChCoAAAALEKoAAAAsQKgCAACwAKEKAADAAoQqAAAACxCqAAAALECoAgAAsAChCgAAwAKEKgAAAAsQqgAAACxAqAIAALAAoQoAAMAChCoAAAALEKou0uLFi9W2bVs5nU4lJCToo48+8ndLAACgASBUXYSVK1cqMzNT06ZN07Zt29S9e3clJyfr8OHD/m4NAAD4WRN/NxBI5s2bp4ceekijRo2SJGVnZ2vdunVaunSpJk+e7Ofu0NgYhqGKigp/twHJ6+/A36ThcDqdstls/m4DjQihqp6qqqpUUFCgrKwsc19QUJCSkpLkcrnqfExlZaUqKyvNbY/Hc8n7RONRUVGhlJQUf7eB7xk0aJC/W8D/l5OTo5CQEH+3gUaEy3/19M0336impkZRUVFe+6OiouR2u+t8zOzZsxUeHm7e4uLifopWAQCAH3Cm6hLKyspSZmamue3xeAhWsIzT6VROTo6/24BOX4o9c1ba4XBwyamBcDqd/m4BjQyhqp5at26t4OBglZSUeO0vKSlRdHR0nY9xOBxyOBw/RXtohGw2G5c2GpBmzZr5uwUAfsblv3qy2+3q3bu38vLyzH21tbXKy8tTYmKiHzsDAAANAWeqLkJmZqZGjhyp6667TjfccIPmz5+vsrIy89uAAACg8SJUXYR7771XR44c0dSpU+V2u9WjRw+tX7/+nMnrAACg8bEZhmH4u4nGwuPxKDw8XKWlpQoLC/N3OwAAoB7q+/nNnCoAAAALEKoAAAAsQKgCAACwAKEKAADAAoQqAAAACxCqAAAALECoAgAAsAChCgAAwAKEKgAAAAvwMzU/oTOL13s8Hj93AgAA6uvM5/aFfoSGUPUT+u677yRJcXFxfu4EAABcrO+++07h4eHnHee3/35CtbW1OnjwoEJDQ2Wz2fzdDgALeTwexcXF6auvvuK3PYHLjGEY+u677xQbG6ugoPPPnCJUAYAF+MF0AExUBwAAsAChCgAAwAKEKgCwgMPh0LRp0+RwOPzdCgA/YU4VAACABThTBQAAYAFCFQAAgAUIVQAAABYgVAEAAFiAUAUAAGABQhUAAIAFCFUAAAAWIFQBAABY4P8Bbm5CN8ePNb0AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(y=df['EstimatedSalary'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "fe5f2107",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: ylabel='CreditScore'>"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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AACAYAkodi655BI9++yzrZY/88wz6tevX4eHAgAACJaATmPdfffd+slPfqLXX3/dumbnrbfe0o4dO74xggAAAOwS0JGd7Oxs7d69WxdccIEKCgpUUFCgCy64QG+//TZfbgcAAEJKwB89Hz58uDZu3BjMWQAAAIIuoCM7r7zyil599dVWy1999VUVFhZ2eCgAAIBgCSh2FixYoKamplbL/X6/FixY0OGhAAAAgiWg2Dl8+LBSU1NbLR8wYICOHDnS4aEAAACCJaDYcblc3/jL5keOHFFcXFyHhwIAAAiWgGJn4sSJmjNnjo4ePWotO3LkiH75y19qwoQJQRsOAACgowKKnfz8fMXFxWnAgAFKSUlRSkqKBg4cqJ49e2rFihXBnhEAACBgAX303OVyqbi4WEVFRdq3b59iY2M1ZMgQXXXVVcGeDwAAoEMC/p4dh8OhcePGady4ccGcBwAAIKjadRqrpKREW7ZsabFsw4YNSklJUUJCgmbMmKG6urqgDggAANAR7YqdJUuWaP/+/db90tJSTZs2Tenp6VqwYIE2b96s5cuXB31IAACAQLUrdt59912NHTvWuv/0009r5MiRevzxxzVv3jytWrWKHwIFAAAhpV2x8/nnn8vtdlv3d+7cqczMTOv+ZZddpvLy8uBNBwAA0EHtih23261jx45Jkurr6/XOO+9o1KhR1vrq6mpFRkYGd0IAAIAOaFfsjB8/XgsWLNAbb7yhvLw8denSRVdeeaW1/r333tPFF18c9CEBAAAC1a6Pni9dulTXX3+9rr76anXt2lXr169XVFSUtf7JJ5/ko+gAACCktCt2LrjgAr3++uvyer3q2rWrwsPDW6x/7rnn1K1bt6AOCAAA0BEB/VzE3Llz9eWXX7ZaHh0drVtuuaXDQwEAAARLQLGzfv161dTUtFpeU1OjDRs2dHgoAACAYGnXaSyfzye/3y+/36/q6mrFxMRY65qamvTKK68oISEh6EMCAAAEql2xEx8fL4fDIYfDof79+7da73A4dPfddwdtOAAAgI5qV+z85S9/kd/v15gxY/T888+rR48e1rqoqCj17dtXSUlJQR8SAAAgUO2KnauvvlqSdOzYMfXp00cOh+OsDAUAABAsbY6d9957T5deeqnCwsLk9XpVWlp6xm2HDBkSlOEAAAA6qs2fxkpLS9Onn35q/T106FClpaW1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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(y=df['CreditScore'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "6ca17dc6",
   "metadata": {},
   "outputs": [],
   "source": [
    "outliers = find_outliers_IQR(df['CreditScore'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "f9fc41c4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n",
      "nan\n",
      "nan\n"
     ]
    }
   ],
   "source": [
    "print(len(outliers))\n",
    "print(outliers.max())\n",
    "print(outliers.min())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "91fd24e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['CreditScore'] = impute_outliers_IQR(df['CreditScore'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "962f286e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    10000.000000\n",
       "mean       650.962593\n",
       "std         96.000144\n",
       "min        383.000000\n",
       "25%        584.000000\n",
       "50%        652.000000\n",
       "75%        718.000000\n",
       "max        850.000000\n",
       "Name: CreditScore, dtype: float64"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()['CreditScore']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "c955f5ab",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: ylabel='CreditScore'>"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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AACAYAkodi655BI9++yzrZY/88wz6tevX4eHAgAACJaATmPdfffd+slPfqLXX3/dumbnrbfe0o4dO74xggAAAOwS0JGd7Oxs7d69WxdccIEKCgpUUFCgCy64QG+//TZfbgcAAEJKwB89Hz58uDZu3BjMWQAAAIIuoCM7r7zyil599dVWy1999VUVFhZ2eCgAAIBgCSh2FixYoKamplbL/X6/FixY0OGhAAAAgiWg2Dl8+LBSU1NbLR8wYICOHDnS4aEAAACCJaDYcblc3/jL5keOHFFcXFyHhwIAAAiWgGJn4sSJmjNnjo4ePWotO3LkiH75y19qwoQJQRsOAACgowKKnfz8fMXFxWnAgAFKSUlRSkqKBg4cqJ49e2rFihXBnhEAACBgAX303OVyqbi4WEVFRdq3b59iY2M1ZMgQXXXVVcGeDwAAoEMC/p4dh8OhcePGady4ccGcBwAAIKjadRqrpKREW7ZsabFsw4YNSklJUUJCgmbMmKG6urqgDggAANAR7YqdJUuWaP/+/db90tJSTZs2Tenp6VqwYIE2b96s5cuXB31IAACAQLUrdt59912NHTvWuv/0009r5MiRevzxxzVv3jytWrWKHwIFAAAhpV2x8/nnn8vtdlv3d+7cqczMTOv+ZZddpvLy8uBNBwAA0EHtih23261jx45Jkurr6/XOO+9o1KhR1vrq6mpFRkYGd0IAAIAOaFfsjB8/XgsWLNAbb7yhvLw8denSRVdeeaW1/r333tPFF18c9CEBAAAC1a6Pni9dulTXX3+9rr76anXt2lXr169XVFSUtf7JJ5/ko+gAACCktCt2LrjgAr3++uvyer3q2rWrwsPDW6x/7rnn1K1bt6AOCAAA0BEB/VzE3Llz9eWXX7ZaHh0drVtuuaXDQwEAAARLQLGzfv161dTUtFpeU1OjDRs2dHgoAACAYGnXaSyfzye/3y+/36/q6mrFxMRY65qamvTKK68oISEh6EMCAAAEql2xEx8fL4fDIYfDof79+7da73A4dPfddwdtOAAAgI5qV+z85S9/kd/v15gxY/T888+rR48e1rqoqCj17dtXSUlJQR8SAAAgUO2KnauvvlqSdOzYMfXp00cOh+OsDAUAABAsbY6d9957T5deeqnCwsLk9XpVWlp6xm2HDBkSlOEAAAA6qs2fxkpLS9Onn35q/T106FClpaW1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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(y=df['CreditScore'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "41cf7910",
   "metadata": {},
   "outputs": [],
   "source": [
    "#8. Split the data into dependent and independent variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "07bf54f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "features=list(set(df)-set(['Exited']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "8a2f3999",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['HasCrCard',\n",
       " 'Gender',\n",
       " 'IsActiveMember',\n",
       " 'Age',\n",
       " 'NumOfProducts',\n",
       " 'Geography',\n",
       " 'CustomerId',\n",
       " 'CreditScore',\n",
       " 'Tenure',\n",
       " 'Balance',\n",
       " 'EstimatedSalary',\n",
       " 'Surname']"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "fe84692a",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=df[features].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "fa2625fc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 1, 1, ..., 0.0, 101348.88, 'Hargrave'],\n",
       "       [0, 1, 1, ..., 83807.86, 112542.58, 'Hill'],\n",
       "       [1, 1, 0, ..., 159660.8, 113931.57, 'Onio'],\n",
       "       ...,\n",
       "       [0, 1, 1, ..., 0.0, 42085.58, 'Liu'],\n",
       "       [1, 0, 0, ..., 75075.31, 92888.52, 'Sabbatini'],\n",
       "       [1, 1, 0, ..., 130142.79, 38190.78, 'Walker']], dtype=object)"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "1dc1d338",
   "metadata": {},
   "outputs": [],
   "source": [
    "y=df['Exited'].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "a6f374bb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0, 1, ..., 1, 1, 0], dtype=int64)"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "966a5366",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>Gender</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>Age</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>Geography</th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Surname</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>RowNumber</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>42</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>15634602</td>\n",
       "      <td>619.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>101348.88</td>\n",
       "      <td>Hargrave</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>15647311</td>\n",
       "      <td>608.0</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>112542.58</td>\n",
       "      <td>Hill</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>42</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>15619304</td>\n",
       "      <td>502.0</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>113931.57</td>\n",
       "      <td>Onio</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>39</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>15701354</td>\n",
       "      <td>699.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>93826.63</td>\n",
       "      <td>Boni</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>43</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>15737888</td>\n",
       "      <td>850.0</td>\n",
       "      <td>2</td>\n",
       "      <td>125510.82</td>\n",
       "      <td>79084.10</td>\n",
       "      <td>Mitchell</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9996</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>39</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>15606229</td>\n",
       "      <td>771.0</td>\n",
       "      <td>5</td>\n",
       "      <td>0.00</td>\n",
       "      <td>96270.64</td>\n",
       "      <td>Obijiaku</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9997</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>35</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>15569892</td>\n",
       "      <td>516.0</td>\n",
       "      <td>10</td>\n",
       "      <td>57369.61</td>\n",
       "      <td>101699.77</td>\n",
       "      <td>Johnstone</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9998</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>36</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>15584532</td>\n",
       "      <td>709.0</td>\n",
       "      <td>7</td>\n",
       "      <td>0.00</td>\n",
       "      <td>42085.58</td>\n",
       "      <td>Liu</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9999</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>15682355</td>\n",
       "      <td>772.0</td>\n",
       "      <td>3</td>\n",
       "      <td>75075.31</td>\n",
       "      <td>92888.52</td>\n",
       "      <td>Sabbatini</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10000</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>15628319</td>\n",
       "      <td>792.0</td>\n",
       "      <td>4</td>\n",
       "      <td>130142.79</td>\n",
       "      <td>38190.78</td>\n",
       "      <td>Walker</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10000 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "           HasCrCard  Gender  IsActiveMember  Age  NumOfProducts  Geography  \\\n",
       "RowNumber                                                                     \n",
       "1                  1       1               1   42              1          0   \n",
       "2                  0       1               1   41              1          2   \n",
       "3                  1       1               0   42              3          0   \n",
       "4                  0       1               0   39              2          0   \n",
       "5                  1       1               1   43              1          2   \n",
       "...              ...     ...             ...  ...            ...        ...   \n",
       "9996               1       0               0   39              2          0   \n",
       "9997               1       0               1   35              1          0   \n",
       "9998               0       1               1   36              1          0   \n",
       "9999               1       0               0   42              2          1   \n",
       "10000              1       1               0   28              1          0   \n",
       "\n",
       "           CustomerId  CreditScore  Tenure    Balance  EstimatedSalary  \\\n",
       "RowNumber                                                                \n",
       "1            15634602        619.0       2       0.00        101348.88   \n",
       "2            15647311        608.0       1   83807.86        112542.58   \n",
       "3            15619304        502.0       8  159660.80        113931.57   \n",
       "4            15701354        699.0       1       0.00         93826.63   \n",
       "5            15737888        850.0       2  125510.82         79084.10   \n",
       "...               ...          ...     ...        ...              ...   \n",
       "9996         15606229        771.0       5       0.00         96270.64   \n",
       "9997         15569892        516.0      10   57369.61        101699.77   \n",
       "9998         15584532        709.0       7       0.00         42085.58   \n",
       "9999         15682355        772.0       3   75075.31         92888.52   \n",
       "10000        15628319        792.0       4  130142.79         38190.78   \n",
       "\n",
       "             Surname  \n",
       "RowNumber             \n",
       "1           Hargrave  \n",
       "2               Hill  \n",
       "3               Onio  \n",
       "4               Boni  \n",
       "5           Mitchell  \n",
       "...              ...  \n",
       "9996        Obijiaku  \n",
       "9997       Johnstone  \n",
       "9998             Liu  \n",
       "9999       Sabbatini  \n",
       "10000         Walker  \n",
       "\n",
       "[10000 rows x 12 columns]"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[features]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "925a2a8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#9. Scale the independent variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "9a307f74",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.02188649, -1.22584767, -0.33295983],\n",
       "       [ 0.21653375,  0.11735002, -0.44754872],\n",
       "       [ 0.2406869 ,  1.33305335, -1.55176894],\n",
       "       ...,\n",
       "       [-1.00864308, -1.22584767,  0.60458564],\n",
       "       [-0.12523071, -0.02260751,  1.26086748],\n",
       "       [-1.07636976,  0.85996499,  1.46921091]])"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scale = StandardScaler()\n",
    "x = scale.fit_transform(df[['EstimatedSalary','Balance','CreditScore']])\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "e48bf0eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "#10. Split the data into training and testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "2f976c78",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "ac9e51f0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([[ 1.10838187, -1.22584767,  0.16706442],\n",
       "        [-0.74759209, -0.01289171, -2.33305684],\n",
       "        [ 1.48746417,  0.57507592, -1.20800227],\n",
       "        ...,\n",
       "        [ 1.41441489,  1.35890908,  0.90668363],\n",
       "        [ 0.84614739, -1.22584767, -0.63505781],\n",
       "        [ 0.32630495,  0.50630343, -0.29129114]]),\n",
       " array([[ 1.61304597,  0.87532296, -0.56213761],\n",
       "        [ 0.49753166,  0.42442221, -1.33300833],\n",
       "        [-0.4235611 ,  0.30292727,  0.57333413],\n",
       "        ...,\n",
       "        [ 0.72065149,  1.29470288, -0.76006388],\n",
       "        [-1.54438254,  1.0563022 , -0.0100275 ],\n",
       "        [ 1.61474887,  0.81611017, -0.81214974]]),\n",
       " array([0, 0, 0, ..., 0, 0, 1], dtype=int64),\n",
       " array([0, 1, 0, ..., 0, 0, 0], dtype=int64))"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_x,test_x,train_y,test_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "70b10545",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
