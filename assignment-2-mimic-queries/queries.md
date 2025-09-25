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

### Meaning

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