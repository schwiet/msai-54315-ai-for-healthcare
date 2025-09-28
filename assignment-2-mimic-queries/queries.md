# Assignment 2 – MIMIC SQL

## Learning Outcomes

After finishing this assignment, you should be able to say: 

- I know how EHR data can be structured in a database.
- I can write SQL queries to retrieve data points about EHR data. 
- I can explain the results of a query.

## Rationale

The goal of this assignment is to get familiar with the EHR data as a database and to use SQL queries to retrieve interesting data points or statistics. It is the next step to understanding data. This assignment will provide an opportunity for you to gain proficiency with important skills related to database and SQL query. 

## Instructions

Develop 10 SQL queries. For full credit, these must include:

1. 1 JOIN table query
2. 1 GROUP BY query
3. 1 nested query (e.g., select is embedded in another select query) using the MIMIC dataset

For this assignment, we recommend setting up MIMIC-III or MIMIC-IV with Google BigQuery (see Tutorial_MIMIC3_BigQuery.pdf), or you can load the databases to your local MySQL Workbench or other database software you prefer. If some tables are too large to load into MySQL or you do not already have access to the MIMIC database, you may use the MIMIC-IV demo dataLinks to an external site. or MIMIC-III demo dataLinks to an external site., whose tables contain only 100 patients. Some join table queries will not yield any results if the demo tables are limited. So, we recommend loading the full tables for patients, admissions, icustays, and other smaller-sized tables. For other large-sized tables (>10MB), you might want to use their demo versions.

Please make your SQL queries interesting and meaningful. Consider the kinds of statistics that might be useful to someone who is doing healthcare research or working in quality improvement. 

Once you have built and executed your queries, create a slide deck with a description, screenshots, and an interpretation of the results for each one. 

If you can align your 10 queries into one cohesive story, you can get one bonus point. For example, telling a story of the descriptive statistics of a certain disease: the distribution of patients by gender, insurance, top medication prescribed, their length of stay range, readmission percentage, major lab tests, and so on.

## Submission

Slide Deck of 10 Queries
- Each query should include the following:
  1. The meaning/description of the query
  2. The SQL query
  3. Screenshots of the results
  4. A _simple_ interpretation of the results
- Optional Bonus:
  - Additional slide telling the overall story of the queries.

## Query 1 - Basic Patient Info

### Description

This basic query of the patient table selects a few of the table's columns, including: `DOB`, `DOD`, `GENDER` and the table's primary key `SUBJECT_ID`. It also derives a new column, `BIRTH_YEAR` from the `DOB` column. Results are ordered by the derived `BIRTH_YEAR`.

### SQL Query

```
SELECT
  `DOB`,
  `DOD`,
  `GENDER`,
  `SUBJECT_ID`,
  EXTRACT(YEAR FROM DOB) AS BIRTH_YEAR
FROM
  `physionet-data.mimiciii_clinical.patients`
ORDER BY BIRTH_YEAR;
```

### Interpretation of Results

The query returns a row for each of the patients in the table. The rows are sorted by birth year in ascending order. Each patiet row includes: date of birth, date of death, gender, the patient's subject ID and the year of their birth.

## Query 2 – Admissions by Year of Birth

### Description

This query counts the number of admissions for patients of each birth year present in the data. It combines the Patients table with the Admissions table to retrieve the DOB for each admission record's patient. Similar to the previous query, year of birth is derived then used to group rows to create a count of admissions for each birth year.

### SQL Query

```
SELECT
  EXTRACT(YEAR FROM Patients.DOB) AS BIRTH_YEAR,
  COUNT(Admissions.HADM_ID) AS admission_count
FROM
  `physionet-data.mimiciii_clinical.patients` AS Patients
JOIN
  `physionet-data.mimiciii_clinical.admissions` AS Admissions
ON
  Patients.SUBJECT_ID = Admissions.SUBJECT_ID
GROUP BY
  BIRTH_YEAR
ORDER BY
  BIRTH_YEAR;
```

### Interpretation of Results

The results list each of the birth years present in the joined table, from 1800 to 2201 in ascending order, accompanied by the number of admissions in the table for patients with that rows' birth year.

> **NOTE**: There is a conspicuous jump between `1901` and `2012` likely explained by the age shift to comply with HIPAA for patients older than 89.

## Query 3 - Admissions by Age

### Description

This query counts the number of admissions for patients of each age present in the data. It also combines the Patients table with the Admissions table to determine the difference between the admission time and DOB.

### SQL Query

```
SELECT
  DATE_DIFF(DATE(Admissions.ADMITTIME), DATE(Patients.DOB), YEAR) AS AGE_AT_ADMISSION,
  COUNT(Admissions.HADM_ID) AS admission_count
FROM
  `physionet-data.mimiciii_clinical.patients` AS Patients
JOIN
  `physionet-data.mimiciii_clinical.admissions` AS Admissions
ON
  Patients.SUBJECT_ID = Admissions.SUBJECT_ID
GROUP BY
  AGE_AT_ADMISSION
ORDER BY
  AGE_AT_ADMISSION;
```

### Interpretation of Results

The results list a row for each age at time of admission present in the joined table, in ascending order. Each age includes a tally for how many admissions occur in the Admissions table for patients of the row's age.

> **NOTE**: There are many admissions for ages above `300`, explained by the age shift to comply with HIPAA for patients older than 89. 

## Query 4 - Confirming Shifted Year of Birth Cohort

### Description

Based on the results of the previous two queries, it appears that any patient with a birth year before `2012` has had their age shifted. This query intends to confirm this assumption by counting the number of patients with a birth year before `2012` who are determined to be less than `300` years old at admission time.

### SQL Query

```
WITH SuspectedShiftPatients AS(
  SELECT
    SUBJECT_ID,
    DOB,
  FROM `physionet-data.mimiciii_clinical.patients`
  WHERE
    EXTRACT(YEAR FROM DOB) < 2012
)
SELECT
  COUNT(Admissions.HADM_ID) AS ADMISSION_COUNT
FROM
  SuspectedShiftPatients
JOIN
  `physionet-data.mimiciii_clinical.admissions` AS Admissions
ON
  SuspectedShiftPatients.SUBJECT_ID = Admissions.SUBJECT_ID
WHERE DATE_DIFF(DATE(Admissions.ADMITTIME), DATE(SuspectedShiftPatients.DOB), YEAR) < 300;
```

### Interpretation of Results

The result is `0`, confirming that all of the patients with birth years prior to `2012` have had their birth year shifted and were thus older than `89` at some point in the dataset. We cannot accurately determine their age because of this shift, we can only say they were older than `89`.

## Query 5 – Admission Count per Age Group

### Description

This query provides a simple breakdown of hospital admission volume in the dataset across different age categories. It categorizes patients’ hospital admissions into groups based on age at admission and counts the number of unique admissions in each group for comparison. It joins the patients and admissions tables so that each admission time can be paired with the patient’s date of birth to calculate age at admission.

### SQL Query

```
WITH ICU_Admissions AS(
  SELECT
    Patients.SUBJECT_ID,
    Admissions.HADM_ID,
    DATE_DIFF(DATE(Admissions.ADMITTIME), DATE(Patients.DOB), YEAR) AS AGE_AT_ADMISSION,
  FROM
    `physionet-data.mimiciii_clinical.patients` Patients
  JOIN
    `physionet-data.mimiciii_clinical.admissions` Admissions USING(SUBJECT_ID)
)
SELECT
  CASE
    WHEN AGE_AT_ADMISSION < 1 THEN 'Neonate (<1)'
    WHEN AGE_AT_ADMISSION BETWEEN 1 AND 24 THEN 'Youth (<25)'
    WHEN AGE_AT_ADMISSION BETWEEN 25 AND 64 THEN 'Adult (26-64)'
    ELSE 'Senior (65+)'
  END AS AGE_GROUP,
  COUNT(DISTINCT HADM_ID) AS ADMISSION_COUNT
FROM
  ICU_Admissions
GROUP BY
  AGE_GROUP
```

### Interpretation of Results

> ***TODO***

## Query 6 – Number of ICU Stays per Age Group

### Description

This query groups ICU stays into age categories and counts the number of stays recorded per stay in each of the age groups defined in the previous query. It first joins patients, admissions, and ICU stay records to compute age at admission, counting stays and then aggregates those counts by each of the age groups.

```
WITH ICU_Admissions AS(
  SELECT
    Patients.SUBJECT_ID,
    Admissions.HADM_ID,
    ICU_Stays.ICUSTAY_ID,
    DATE_DIFF(DATE(Admissions.ADMITTIME), DATE(Patients.DOB), YEAR) AS AGE_AT_ADMISSION,
  FROM
    `physionet-data.mimiciii_clinical.patients` Patients
  JOIN
    `physionet-data.mimiciii_clinical.admissions` Admissions USING(SUBJECT_ID)
  JOIN
    `physionet-data.mimiciii_clinical.icustays` ICU_Stays USING (SUBJECT_ID, HADM_ID)
)
SELECT
  CASE
    WHEN AGE_AT_ADMISSION < 1 THEN 'Neonate (<1)'
    WHEN AGE_AT_ADMISSION BETWEEN 1 AND 24 THEN 'Youth (<25)'
    WHEN AGE_AT_ADMISSION BETWEEN 25 AND 64 THEN 'Adult (26-64)'
    ELSE 'Senior (65+)'
  END AS AGE_GROUP,
  COUNT(DISTINCT ICUSTAY_ID) AS N_STAYS,
FROM
  ICU_Admissions
GROUP BY
  AGE_GROUP
ORDER BY
  CASE AGE_GROUP
    WHEN 'Neonate (<1)' THEN 1
    WHEN 'Youth (<25)' THEN 2
    WHEN 'Adult (26-64)' THEN 3
    WHEN 'Senior (65+)' THEN 4
    ELSE 5
  END;
```

### Interpretation of Results

> ***TODO***

### Query 7 – ICU Stays per Admission by Age Group

### Description

After noticing that ICU Stay counts where higher per age group than corresponding Admission counts, this query shows the average and max ICU stays per admission, again grouped by age range.

### SQL Query

```
WITH ICU_Admissions AS(
  SELECT
    Patients.SUBJECT_ID,
    Admissions.HADM_ID,
    ICU_Stays.ICUSTAY_ID,
    DATE_DIFF(DATE(Admissions.ADMITTIME), DATE(Patients.DOB), YEAR) AS AGE_AT_ADMISSION,
  FROM
    `physionet-data.mimiciii_clinical.patients` Patients
  JOIN
    `physionet-data.mimiciii_clinical.admissions` Admissions USING(SUBJECT_ID)
  JOIN
    `physionet-data.mimiciii_clinical.icustays` ICU_Stays USING (SUBJECT_ID, HADM_ID)
),
StaysPerAdmission AS (
  SELECT
    SUBJECT_ID,
    HADM_ID,
    AGE_AT_ADMISSION,
    COUNT(DISTINCT ICUSTAY_ID) AS ICU_STAY_COUNT
  FROM
    ICU_Admissions
  GROUP BY
    SUBJECT_ID,
    HADM_ID,
    AGE_AT_ADMISSION
)
SELECT
  CASE
    WHEN AGE_AT_ADMISSION < 1 THEN 'Neonate (<1)'
    WHEN AGE_AT_ADMISSION BETWEEN 1 AND 24 THEN 'Youth (<25)'
    WHEN AGE_AT_ADMISSION BETWEEN 25 AND 64 THEN 'Adult (26–64)'
    ELSE 'Senior (65+)'
  END AS AGE_GROUP,
  COUNT(DISTINCT HADM_ID) AS N_ADMISSIONS,
  ROUND(AVG(ICU_STAY_COUNT), 2) AS AVG_ICU_STAYS_PER_ADMISSION,
  MAX(ICU_STAY_COUNT) AS MAX_STAYS_PER_ADMISSIONS
FROM StaysPerAdmission
GROUP BY AGE_GROUP
ORDER BY
  CASE AGE_GROUP
    WHEN 'Neonate (<1)'   THEN 1
    WHEN 'Youth (<25)'    THEN 2
    WHEN 'Adult (26–64)'  THEN 3
    WHEN 'Senior (65+)'   THEN 4
  END;
```

### Interpretation of Results

Each age group's row has the average and max number of ICU Stays per admission. Interestingly, each age group has an average slightly greater than `1`.

## Query 8 – Average Procedure Count per Age Group

### Description

Building off of the previous query, this query groups ICU stays into age categories and calculates the _average number of procedure codes_ recorded per stay in each of the same age groups. Again, it joins patients, admissions, and ICU stay records to compute age at admission. Nex it adds a new table, the [ICD9 PROCEDURES](https://mimic.mit.edu/docs/iii/tables/procedures_icd/) table to count procedures per stay. Finally, it aggregates those counts by each of the age groups, displaying the average.

### Query

```
WITH ICU_Admissions AS(
  SELECT
    Patients.SUBJECT_ID,
    Admissions.HADM_ID,
    ICU_Stays.ICUSTAY_ID,
    DATE_DIFF(DATE(Admissions.ADMITTIME), DATE(Patients.DOB), YEAR) AS AGE_AT_ADMISSION,
  FROM
    `physionet-data.mimiciii_clinical.patients` Patients
  JOIN
    `physionet-data.mimiciii_clinical.admissions` Admissions USING(SUBJECT_ID)
  JOIN
    `physionet-data.mimiciii_clinical.icustays` ICU_Stays USING (SUBJECT_ID, HADM_ID)
),
ProcedureCounts AS (
  SELECT
    ICU_Admissions.SUBJECT_ID,
    ICU_Admissions.HADM_ID,
    ICU_Admissions.ICUSTAY_ID,
    ICU_Admissions.AGE_AT_ADMISSION,
    COUNT(Procedures.SEQ_NUM) AS PROC_COUNT
  FROM ICU_Admissions
  LEFT JOIN `physionet-data.mimiciii_clinical.procedures_icd` Procedures
    ON ICU_Admissions.SUBJECT_ID = Procedures.SUBJECT_ID
   AND ICU_Admissions.HADM_ID = Procedures.HADM_ID
  GROUP BY
    ICU_Admissions.SUBJECT_ID,
    ICU_Admissions.HADM_ID,
    ICU_Admissions.ICUSTAY_ID,
    ICU_Admissions.AGE_AT_ADMISSION
)
SELECT
  CASE
    WHEN AGE_AT_ADMISSION < 1 THEN 'Neonate (<1)'
    WHEN AGE_AT_ADMISSION BETWEEN 1 AND 24 THEN 'Youth (<25)'
    WHEN AGE_AT_ADMISSION BETWEEN 25 AND 64 THEN 'Adult (26-64)'
    ELSE 'Senior (65+)'
  END AS AGE_GROUP,
  ROUND(AVG(PROC_COUNT), 2) AS AVG_PROCS_PER_STAY
  APPROX_QUANTILES(PROC_COUNT, 101)[OFFSET(50)] AS MEDIAN_PROC_COUNT,
  MAX(PROC_COUNT) AS MAX_PROC_COUNT
FROM
  ProcedureCounts
GROUP BY
  AGE_GROUP
ORDER BY
  CASE AGE_GROUP
    WHEN 'Neonate (<1)' THEN 1
    WHEN 'Youth (<25)' THEN 2
    WHEN 'Adult (26-64)' THEN 3
    WHEN 'Senior (65+)' THEN 4
    ELSE 5
  END;
```

### Interpretation of Results

Each of the groups are listed with the mean, median and max procedure count. As might be expected, the number of procedures increases with age, though the mean for the 26-64 group is slightly highest, due to outliers at the upper extreme.

## Query 9 - ICU Stay Time Info by Age Group

### Description

This query combines the Patient, Admissions and ICUStays tables to continue to analyze the relationship of a patients age with other data in the dataset. In this case, continuing to use the same age groups, it compares the lengths of stay between groups.

### SQL Query

```
WITH ICU_Admissions AS (
  SELECT
    p.SUBJECT_ID,
    a.HADM_ID,
    i.ICUSTAY_ID,
    i.INTIME,
    i.OUTTIME,
    DATE_DIFF(DATE(a.ADMITTIME), DATE(p.DOB), YEAR) AS AGE_AT_ADMISSION,
    TIMESTAMP_DIFF(i.OUTTIME, i.INTIME, HOUR) / 24.0 AS STAY_DAYS
  FROM `physionet-data.mimiciii_clinical.patients`   AS p
  JOIN `physionet-data.mimiciii_clinical.admissions` AS a USING (SUBJECT_ID)
  JOIN `physionet-data.mimiciii_clinical.icustays`   AS i USING (SUBJECT_ID, HADM_ID)
  WHERE i.INTIME IS NOT NULL AND i.OUTTIME IS NOT NULL
),
ValidStays AS (
  SELECT
    ICUSTAY_ID,
    HADM_ID,
    STAY_DAYS,
    CASE
      WHEN AGE_AT_ADMISSION < 1 THEN 'Neonate (<1)'
      WHEN AGE_AT_ADMISSION BETWEEN 1 AND 24 THEN 'Youth (<25)'
      WHEN AGE_AT_ADMISSION BETWEEN 25 AND 64 THEN 'Adult (26-64)'
      ELSE 'Senior (65+)'
    END AS AGE_GROUP
  FROM ICU_Admissions
  WHERE STAY_DAYS >= 0
)
SELECT
  AGE_GROUP,
  COUNT(DISTINCT ICUSTAY_ID)                                    AS N_STAYS,
  ROUND(AVG(STAY_DAYS), 2)                                      AS MEAN_STAY_DAYS,
  ROUND(APPROX_QUANTILES(STAY_DAYS, 101)[OFFSET(50)], 2)        AS MEDIAN_STAY_DAYS,
  ROUND(MAX(STAY_DAYS), 2)                                      AS MAX_STAY_DAYS
FROM ValidStays
GROUP BY AGE_GROUP
ORDER BY
  CASE AGE_GROUP
    WHEN 'Neonate (<1)'  THEN 1
    WHEN 'Youth (<25)'   THEN 2
    WHEN 'Adult (26-64)' THEN 3
    WHEN 'Senior (65+)'  THEN 4
  END;
```

### Interpretation of Results

For each group, the Mean, Median and Max stay time are listed. The Max values highlight outliers that affect the Means in each age group. Intuitively, as age increases, median stay time increases.

## Query 10 – Delay from Admission to ICU Transfer by Age

### Description

This query determines how long it takes patients to reach the ICU after initial admission and summarizes the results by age group. The Patients, Admissions and ICU Stays tables are joined to gather all necessary columns.

### SQL Query

```
WITH ICU_Admissions AS (
  SELECT
    Patients.SUBJECT_ID,
    Admissions.HADM_ID,
    ICU_Stays.ICUSTAY_ID,
    Admissions.ADMITTIME,
    ICU_Stays.INTIME,
    DATE_DIFF(DATE(Admissions.ADMITTIME), DATE(Patients.DOB), YEAR) AS AGE_AT_ADMISSION,
    TIMESTAMP_DIFF(ICU_Stays.INTIME, Admissions.ADMITTIME, MINUTE) / 60.0 AS HOURS_TO_ICU
  FROM `physionet-data.mimiciii_clinical.patients`   Patients
  JOIN `physionet-data.mimiciii_clinical.admissions` Admissions USING (SUBJECT_ID)
  JOIN `physionet-data.mimiciii_clinical.icustays`   ICU_Stays USING (SUBJECT_ID, HADM_ID)
  WHERE Admissions.ADMITTIME IS NOT NULL AND ICU_STAYS.INTIME IS NOT NULL
),
Valid AS (
  SELECT
    ICUSTAY_ID,
    CASE
      WHEN AGE_AT_ADMISSION < 1 THEN 'Neonate (<1)'
      WHEN AGE_AT_ADMISSION BETWEEN 1 AND 24 THEN 'Youth (<25)'
      WHEN AGE_AT_ADMISSION BETWEEN 25 AND 64 THEN 'Adult (26-64)'
      ELSE 'Senior (65+)'
    END AS AGE_GROUP,
    HOURS_TO_ICU
  FROM ICU_Admissions
)
SELECT
  AGE_GROUP,
  COUNT(DISTINCT ICUSTAY_ID) AS N_STAYS,
  ROUND(AVG(HOURS_TO_ICU), 1) AS MEAN_HOURS_TO_ICU,
  ROUND(APPROX_QUANTILES(HOURS_TO_ICU, 101)[OFFSET(50)], 1) AS MEDIAN_HOURS_TO_ICU,
  ROUND(MAX(HOURS_TO_ICU), 1) AS MAX_HOURS_TO_ICU,
  ROUND(100 * AVG(CASE WHEN HOURS_TO_ICU <= 6 THEN 1 ELSE 0 END), 1) AS PCT_WITHIN_6H
FROM Valid
GROUP BY AGE_GROUP
ORDER BY
  CASE AGE_GROUP
    WHEN 'Neonate (<1)'  THEN 1
    WHEN 'Youth (<25)'   THEN 2
    WHEN 'Adult (26-64)' THEN 3
    WHEN 'Senior (65+)'  THEN 4
  END;
```

### Interpretation of Results

Each age group is listed with the number of admissions, along with the mean, median, maximum time and percentage of transfers took place within 6 hours from hospital admission. Most transfers, across all age groups, happen within an hour, but there are some extreme outliers in the dataset.