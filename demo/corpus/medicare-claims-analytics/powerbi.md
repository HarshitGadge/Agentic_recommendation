# Building the dashboard in Power BI

The repo ships Power BI-ready tables in `exports/`. You can build the report without running dbt.

## 1. Load the tables

**Get data → Parquet** (or **Folder** pointed at `exports/`), then load:

| Table | Grain | Rows |
|---|---|---|
| `fct_member_years.parquet` | beneficiary × year | 343,644 |
| `fct_inpatient_stays.parquet` | hospital stay | 64,365 |
| `dim_beneficiaries.parquet` | beneficiary | 116,352 |

Add a year dimension with **Modeling → New table**:

```DAX
Dim Year = DATATABLE("Year", INTEGER, {{2008}, {2009}, {2010}})
```

## 2. Relationships (star schema)

| From (many) | To (one) | Filter direction |
|---|---|---|
| `fct_member_years[bene_id]` | `dim_beneficiaries[bene_id]` | single |
| `fct_inpatient_stays[bene_id]` | `dim_beneficiaries[bene_id]` | single |
| `fct_member_years[coverage_year]` | `Dim Year[Year]` | single |
| `fct_inpatient_stays[discharge_year]` | `Dim Year[Year]` | single |

Use `Dim Year[Year]` for the year slicer so it filters both fact tables.

## 3. Measures

```DAX
Members =
CALCULATE ( COUNTROWS ( fct_member_years ), fct_member_years[member_months] > 0 )

Member Months =
CALCULATE ( SUM ( fct_member_years[member_months] ), fct_member_years[member_months] > 0 )

Medicare Paid =
CALCULATE ( SUM ( fct_member_years[medicare_paid_total] ), fct_member_years[member_months] > 0 )

PMPM = DIVIDE ( [Medicare Paid], [Member Months] )

PMPM Inpatient =
DIVIDE ( CALCULATE ( SUM ( fct_member_years[medicare_paid_ip] ), fct_member_years[member_months] > 0 ), [Member Months] )

PMPM Outpatient =
DIVIDE ( CALCULATE ( SUM ( fct_member_years[medicare_paid_op] ), fct_member_years[member_months] > 0 ), [Member Months] )

PMPM Professional =
DIVIDE ( CALCULATE ( SUM ( fct_member_years[medicare_paid_carrier] ), fct_member_years[member_months] > 0 ), [Member Months] )

Admits per 1,000 =
DIVIDE (
    CALCULATE ( SUM ( fct_member_years[inpatient_admissions] ), fct_member_years[member_months] > 0 ) * 12000,
    [Member Months]
)

Index Stays = CALCULATE ( COUNTROWS ( fct_inpatient_stays ), fct_inpatient_stays[is_index_stay] = TRUE () )

Readmissions 30d = CALCULATE ( COUNTROWS ( fct_inpatient_stays ), fct_inpatient_stays[is_readmitted_30d] = TRUE () )

Readmission Rate 30d = DIVIDE ( [Readmissions 30d], [Index Stays] )

Top 5% Spend Share =
VAR Covered = FILTER ( fct_member_years, fct_member_years[member_months] > 0 )
VAR TopN_Count = ROUND ( COUNTROWS ( Covered ) * 0.05, 0 )
VAR TopMembers = TOPN ( TopN_Count, Covered, fct_member_years[medicare_paid_total], DESC )
RETURN
    DIVIDE (
        SUMX ( TopMembers, fct_member_years[medicare_paid_total] ),
        SUMX ( Covered, fct_member_years[medicare_paid_total] )
    )
```

Select a single year before reading `Top 5% Spend Share`; it ranks members within the current filter context.
It should match `agg_cost_concentration` (top 1% + next 4%) to within rounding, since both rank the covered member-years in the year.

To break readmissions down by the patient's chronic burden (page 3), add a calculated column to `fct_inpatient_stays`:

```DAX
Chronic Conditions =
LOOKUPVALUE (
    fct_member_years[chronic_condition_count],
    fct_member_years[bene_id], fct_inpatient_stays[bene_id],
    fct_member_years[coverage_year], fct_inpatient_stays[discharge_year]
)
```

## 4. Suggested pages

1. **Overview:** year slicer; cards for members, PMPM, admits per 1,000, readmission rate, top 5% spend share;
   clustered columns of PMPM by setting by year.
2. **Chronic conditions:** column chart of PMPM by `chronic_condition_count` (group 6+ with a calculated column);
   bar chart of PMPM with vs without each condition (unpivot the `has_*` columns in Power Query).
3. **Readmissions:** readmission rate by `condition_group` and by chronic-condition count of the patient;
   matrix of index stays and readmissions.
4. **Geography:** filled map on `dim_beneficiaries[state_name]` with PMPM and readmission rate as tooltips. This uses each beneficiary's
   most recent state; `exports/agg_state_year.csv` uses the state recorded in each year, so values can differ slightly.

## 5. Check your numbers

After building the cards, compare them with `exports/agg_utilization_by_year.csv` and `exports/agg_readmissions.csv`.
For 2009 you should see 108,555 members (the `Members` measure), PMPM $378, 227 admissions per 1,000 and a 7.4% readmission rate.
