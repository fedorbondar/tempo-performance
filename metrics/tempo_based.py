from datetime import datetime, timedelta
import pandas as pd

from utils.data_builder import DATETIME_FORMAT, START_DAY_TIME, END_DAY_TIME

SUPPORT_ISSUE_TYPES = ['Bug', 'Bug from CLM', 'Defect', 'Error', 'Incident', 'Accident']
ABSENCE_ISSUE_KEYS = ['отпуск', 'отгул', 'отсутствия']


def check_absence_keys_in_issue(issue_summary: str):
    for key in ABSENCE_ISSUE_KEYS:
        if key in issue_summary.lower():
            return True
    return False


def get_supposed_work_hours_by_period(date_from: str, date_until: str, full_time: bool = True):
    datetime_from = datetime.strptime(date_from + START_DAY_TIME, DATETIME_FORMAT).date()
    datetime_until = datetime.strptime(date_until + END_DAY_TIME, DATETIME_FORMAT).date()
    delta = datetime_until - datetime_from

    days_range = [datetime_from + timedelta(days=x) for x in range(delta.days + 1)]
    work_days = sum([1 if day.weekday() < 5 else 0 for day in days_range])
    return work_days * 8.0 if full_time else work_days * 6.0


def compute_initiative_completion_rate(data: pd.DataFrame, date_from: str, date_until: str):
    max_hours_initiative = data.groupby('issuekey').hour.agg('sum').max()
    return 1 - max_hours_initiative / get_supposed_work_hours_by_period(date_from, date_until)


def compute_support_tasks_rate(data: pd.DataFrame, date_from: str, date_until: str):
    tasks_hours_agg = data.groupby('issuekey').agg({'hour': 'sum', 'issue_type': 'first'})
    hours_support_issues = tasks_hours_agg.loc[tasks_hours_agg.issue_type.isin(SUPPORT_ISSUE_TYPES)].hour.sum()
    return 1 - hours_support_issues / get_supposed_work_hours_by_period(date_from, date_until)


def compute_absent_rate(data: pd.DataFrame, date_from: str, date_until: str):
    tasks_hours_agg = data.groupby('issuekey').agg({'hour': 'sum', 'issue_summary': 'first'})
    absence_hours = tasks_hours_agg.loc[tasks_hours_agg.issue_summary.apply(check_absence_keys_in_issue)].hour.sum()
    return 1 - absence_hours / get_supposed_work_hours_by_period(date_from, date_until)


def compute_initiative_share_by_domain(data_author: pd.DataFrame, data_domain: pd.DataFrame):
    return len(data_author.issuekey.unique()) / len(data_domain.issuekey.unique())
