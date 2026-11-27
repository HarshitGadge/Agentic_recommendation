# Metric definitions

All metrics are computed in the dbt marts (`dbt/models/marts/`). Years are calendar years.

| Metric | Definition | Where |
|---|---|---|
| Member months | `max(Part A months, Part B months)` per beneficiary-year, from the beneficiary summary file. Member-years with 0 months are excluded from rate denominators. | `fct_member_years.member_months` |
| Medicare paid | Annual Medicare fee-for-service reimbursement from the beneficiary summary file, inpatient + outpatient + carrier (professional). Beneficiary cost sharing and other primary payers are excluded. | `fct_member_years.medicare_paid_total` |
| PMPM | Medicare paid ÷ member months, for the members in the slice. | `agg_utilization_by_year.pmpm_*` |
| Admissions per 1,000 | Hospital stays admitted in the year × 12,000 ÷ member months (per 1,000 member-years). | `agg_utilization_by_year.admits_per_1000` |
| Inpatient days per 1,000 | Sum of (discharge date − admission date) × 12,000 ÷ member months. | `agg_utilization_by_year` |
| Outpatient claims per 1,000 | Outpatient institutional claims (segments combined) × 12,000 ÷ member months. | `agg_utilization_by_year` |
| Hospital stay | Inpatient claims for one beneficiary merged when they overlap, when the next admission is on the previous discharge date, or when it is the next calendar day at a different provider (a transfer). A next-day admission to the same provider is a new stay. | `int_inpatient_stays` |
| Index stay | A stay discharged alive, with at least 30 days between discharge and the end of the data (2010-12-31). | `fct_inpatient_stays.is_index_stay` |
| 30-day readmission | An index stay followed by another stay admitted 1–30 days after discharge. All-cause: any diagnosis counts. | `fct_inpatient_stays.is_readmitted_30d` |
| Readmission rate | Readmissions ÷ index stays, by discharge year. | `agg_readmissions` |
| Condition group | Principal diagnosis of the first claim in the stay mapped to simplified CMS HRRP ICD-9 cohorts (AMI, heart failure, pneumonia, COPD, other). Code lists are in `dbt/macros/readmission_condition_group.sql`. | `fct_inpatient_stays.condition_group` |
| Chronic condition count | Number of the 11 CMS Chronic Condition Warehouse flags on the beneficiary summary file set to "yes" that year. | `fct_member_years.chronic_condition_count` |
| Spend band | Members with at least one coverage month, ranked by Medicare paid within a year: top 1%, next 4%, next 15%, next 30%, bottom 50%. | `agg_cost_concentration` |

## Differences from the official CMS measures

The readmission logic follows the structure of the CMS hospital-wide and HRRP measures (transfer chains collapsed, in-hospital deaths
and discharges without 30 days of follow-up excluded, 30-day window) but is simpler:

- no planned-readmission algorithm: every readmission counts;
- no age restriction: the CMS measures cover patients 65 and older, while here about 17.5% of 2008 index stays are for younger
  beneficiaries (entitled through disability or ESRD);
- a stay that is itself a readmission can also be an index stay, as in the hospital-wide measure; the condition-specific measures don't allow this;
- the condition cohort comes from the principal diagnosis of the first claim in a transfer chain, using principal-diagnosis code lists only;
- no exclusions for discharges against medical advice or for missing enrollment in the 30 days after discharge;
- no risk adjustment.
