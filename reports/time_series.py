from utils.data_builder import DataBuilder
import matplotlib.pyplot as plt


def simple_tempo_worklog_report(db: DataBuilder, author: str, date_from: str, date_until: str,
                                ignore_weekends: bool = False):
    series = db.create_series_logged_time(author, date_from, date_until, ignore_weekends)
    plot = series.plot(kind='bar', title="Worklog tempo of " + author)
    fig = plot.get_figure()
    fig.savefig(f"simple_tempo_worklog_{author}.png")
