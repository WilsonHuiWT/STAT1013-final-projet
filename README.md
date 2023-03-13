# STAT1013-final-projet
Here is the final project of STAT1013
---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

<div class="cell markdown" id="9xZnRXM7x0Cv">

# CUHK-STAT1013: Practical Assignment Part 1: Sharing Your Idea and Data

</div>

<div class="cell markdown" id="9Fy05KAkyJI0">

## Graduate Admission dataset background

**Background and basic description of the dataset**:

Dataset describing the chance of admit of Masters Programs of individual
candidates on graduate admission.This dataset was built with the purpose
of helping students in shortlisting universities with their profiles.
The predicted output gives student a fair idea about their chances for a
particular university.This dataset which is inspired by the UCLA
Graduate Dataset is owned by Mohan S Acharya.

The dataset contains serial number and 8 parameters which are considered
influential during the application for Masters Programs.

**Github**:
<https://github.com/prasertcbs/basic-dataset/blob/master/graduate-admissions/Admission_Predict.csv>

**Sample size**: 400

**Feature documentation**:

| Feature           | Class  | Shape  | Dtype   |
|:------------------|:-------|:-------|:--------|
| Serial No.        | Tensor | 1\~400 | int64   |
| GRE Score         | Tensor | 0\~340 | int64   |
| TOEFL Score       | Tensor | 0\~120 | int64   |
| University Rating | Tensor | 1\~5   | int64   |
| SOP               | Tensor | 1\~5   | float64 |
| LOR               | Tensor | 1\~5   | float64 |
| CGPA              | Tensor | 1\~10  | float64 |
| Research          | Tensor | 0/1    | int64   |
| Chance of Admit   | Tensor | 0\~1   | float64 |

</div>

<div class="cell markdown" id="k85zO7zxys4H">

## Hypothesis

-   Tell us what your idea is and why you have chosen to pursue this
    idea.
    -   We are interested in "*Do the student with research experience
        have a greater chance of admit on the Master Programs?*"
-   What two groups you are comparing:
    -   **G1**: chance of admit of student with research experience;
        **G2**: chance of admit of student without research experience
-   What you will be measuring (i.e., what your response variable will
    be)
    -   `chance of admit`
-   Is your response variable quantitative rather than categorical?
    -   `chance` is floating data, with the range from 0 to 1 , which
        can be regarded as a quantitative variable.
-   Make a prediction about what kind of difference you expect to see
    between your samples and WHY.
    -   We'd expect that **G1** \> **G2** since [research experience
        should provide advantage to graduate
        admission](https://research.uoregon.edu/plan/undergraduate-research/resources/benefits-undergraduate-research).
-   Talk about how you will gather your data
    -   From Github link:
        <https://github.com/prasertcbs/basic-dataset/blob/master/graduate-admissions/Admission_Predict.csv>
-   If you had unlimited resources (time, money, staff, etc.) how would
    you collect your data?
    -   \(i\) Attempt to collect more data from different perspective
        since this dataset is created for prediction of Graduate
        Admissions from an Indian perspective only ; (ii) investigate if
        the provided dataset is using simple random sampling or other
        probability sampling method to draw the subset of the original
        graduate population.

</div>

<div class="cell markdown" id="3GOdPWT03PQB">

## Prepare your dataset

</div>

<div class="cell code" execution_count="3"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:206}"
id="mUxJb4hxvpHQ" outputId="b1c8f043-819c-418c-db27-abe9cae6f5d2">

``` python
## load dataset from github

import pandas as pd


df = pd.read_csv('https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/graduate-admissions/Admission_Predict.csv')
df.head(5)
```

<div class="output execute_result" execution_count="3">

       Serial No.  GRE Score  TOEFL Score  University Rating  SOP  LOR   CGPA  \
    0           1        337          118                  4  4.5   4.5  9.65   
    1           2        324          107                  4  4.0   4.5  8.87   
    2           3        316          104                  3  3.0   3.5  8.00   
    3           4        322          110                  3  3.5   2.5  8.67   
    4           5        314          103                  2  2.0   3.0  8.21   

       Research  Chance of Admit   
    0         1              0.92  
    1         1              0.76  
    2         1              0.72  
    3         1              0.80  
    4         0              0.65  

</div>

</div>

<div class="cell markdown" id="55xAIxVa3hpQ">

-   Tell us what groups you want to compare in the dataset
    -   **G1** (Chance of Admit \| Research = 1) vs. **G2** (Chance of
        Admit \| Research = 0)

</div>

<div class="cell markdown" id="13PdL3ht3902">

-   Print first 5 records of each group, respectively.

</div>

<div class="cell code" execution_count="11"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="UNL0WXav3hLj" outputId="1d3efb59-9564-4bc2-8e75-766915db4d67">

``` python
## First 5 records of G1 (male)
(df[df['Research'] == 1]['Chance of Admit ']).head(5)
```

<div class="output execute_result" execution_count="11">

    0    0.92
    1    0.76
    2    0.72
    3    0.80
    5    0.90
    Name: Chance of Admit , dtype: float64

</div>

</div>

<div class="cell code" execution_count="10"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="dhe52HVB4T1O" outputId="678ad4b8-924c-4573-a0cf-6598b3e9b6d1">

``` python
## First 5 records of G2 (No research experience)
(df[df['Research'] == 0]['Chance of Admit ']).head(5)
```

<div class="output execute_result" execution_count="10">

    4     0.65
    7     0.68
    8     0.50
    9     0.45
    15    0.54
    Name: Chance of Admit , dtype: float64

</div>

</div>

<div class="cell markdown" id="zwbMVRLd7Dam">

-   Data description and visualization

</div>

<div class="cell code" execution_count="28"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="3s9uw5fD95kJ" outputId="a85c4b9a-8aeb-482b-db73-199736a1d8d3">

``` python
df.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 400 entries, 0 to 399
    Data columns (total 9 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Serial No.         400 non-null    int64  
     1   GRE Score          400 non-null    int64  
     2   TOEFL Score        400 non-null    int64  
     3   University Rating  400 non-null    int64  
     4   SOP                400 non-null    float64
     5   LOR                400 non-null    float64
     6   CGPA               400 non-null    float64
     7   Research           400 non-null    int64  
     8   Chance of Admit    400 non-null    float64
    dtypes: float64(4), int64(5)
    memory usage: 28.2 KB

</div>

</div>

<div class="cell markdown" id="hPos3T8L98a9">

-   no missing value in the dataset
-   8 quantitative variables

</div>

<div class="cell code" execution_count="22"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:331}"
id="aGahkQWbyIBY" outputId="075f84ed-bbc4-4614-fe02-48d30541bf5d">

``` python
df.describe().T
```

<div class="output execute_result" execution_count="22">

                       count        mean         std     min     25%     50%  \
    Serial No.         400.0  200.500000  115.614301    1.00  100.75  200.50   
    GRE Score          400.0  316.807500   11.473646  290.00  308.00  317.00   
    TOEFL Score        400.0  107.410000    6.069514   92.00  103.00  107.00   
    University Rating  400.0    3.087500    1.143728    1.00    2.00    3.00   
    SOP                400.0    3.400000    1.006869    1.00    2.50    3.50   
    LOR                400.0    3.452500    0.898478    1.00    3.00    3.50   
    CGPA               400.0    8.598925    0.596317    6.80    8.17    8.61   
    Research           400.0    0.547500    0.498362    0.00    0.00    1.00   
    Chance of Admit    400.0    0.724350    0.142609    0.34    0.64    0.73   

                            75%     max  
    Serial No.         300.2500  400.00  
    GRE Score          325.0000  340.00  
    TOEFL Score        112.0000  120.00  
    University Rating    4.0000    5.00  
    SOP                  4.0000    5.00  
    LOR                  4.0000    5.00  
    CGPA                 9.0625    9.92  
    Research             1.0000    1.00  
    Chance of Admit      0.8300    0.97  

</div>

</div>

<div class="cell code" execution_count="27"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:279}"
id="ZljNJtK7z4ma" outputId="39e8a229-be0a-4cd8-942d-45d6ac49e76a">

``` python
import seaborn as sns
import matplotlib.pyplot as plt
sns.violinplot(data=df, x="Research", y="Chance of Admit ")
plt.show()
```

<div class="output display_data">

![](b30232bffba7055907c1cb7887cece8302efc06a.png)

</div>

</div>

<div class="cell code" execution_count="29"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="FG3IrVLn8-NU" outputId="30a36bb2-fb39-496c-9222-d1b8ce289807">

``` python
print(df.groupby('Research')['Chance of Admit '].mean())
```

<div class="output stream stdout">

    Research
    0    0.637680
    1    0.795982
    Name: Chance of Admit , dtype: float64

</div>

</div>

<div class="cell markdown" id="P8_gp2Vq-qKZ">

-   The average chance of admit of student with research experience is
    greater.

</div>
