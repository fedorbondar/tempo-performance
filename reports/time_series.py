from utils.data_builder import DataBuilder
import matplotlib.pyplot as plt


def simple_tempo_worklog_report(db: DataBuilder, author: str, date_from: str, date_until: str,
                                ignore_weekends: bool = False):
    """
    Visualization of physical `author`'s worklog.
    :param db: DataBuilder class object.
    :param author: login of worker who logged time.
    :param date_from: start date in period.
    :param date_until: end date in period.
    :param ignore_weekends: whether to ignore logged time during weekends or not.
    """
    series = db.create_series_logged_time(author, date_from, date_until, ignore_weekends)
    plot = series.plot(kind='bar', title="Worklog tempo of " + author, ylabel='Hours')
    plot.yaxis.grid(True, which='major')
    fig = plot.get_figure()
    fig.savefig(f"simple_tempo_worklog_{author}.png")


def horizontal_tempo_worklog_report(db: DataBuilder, author: str, date_from: str, date_until: str,
                                    ignore_weekends: bool = False):
    """
    Visualization of physical `author`'s worklog with horizontal bars.
    :param db: DataBuilder class object.
    :param author: login of worker who logged time.
    :param date_from: start date in period.
    :param date_until: end date in period.
    :param ignore_weekends: whether to ignore logged time during weekends or not.
    """
    series = db.create_series_logged_time(author, date_from, date_until, ignore_weekends)
    plot = series.plot.barh(title="Worklog tempo of " + author, xlabel='Hours', figsize=(8, 11))
    plot.xaxis.grid(True, which='major')
    fig = plot.get_figure()
    fig.savefig(f"horizontal_tempo_worklog_{author}.png")
