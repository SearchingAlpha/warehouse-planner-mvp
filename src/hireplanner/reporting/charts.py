"""Chart builders for Excel reports using openpyxl."""
from openpyxl.chart import LineChart, BarChart, AreaChart, Reference
from openpyxl.chart.series import SeriesLabel
from openpyxl.utils import get_column_letter


def create_forecast_chart(ws, min_row, max_row, date_col, p10_col, p50_col, p90_col, title="Volume Forecast"):
    """Create a line chart with P10/P50/P90 bands."""
    chart = LineChart()
    chart.title = title
    chart.style = 10
    chart.y_axis.title = "Volume (units)"
    chart.x_axis.title = "Date"
    chart.width = 24
    chart.height = 14

    dates = Reference(ws, min_col=date_col, min_row=min_row + 1, max_row=max_row)

    for col, name, color in [
        (p10_col, "P10", "BDD7EE"),
        (p50_col, "Forecast", "2E75B6"),
        (p90_col, "P90", "BDD7EE"),
    ]:
        data = Reference(ws, min_col=col, min_row=min_row, max_row=max_row)
        chart.add_data(data, titles_from_data=True)
        series = chart.series[-1]
        series.graphicalProperties.line.solidFill = color
        if name != "Forecast":
            series.graphicalProperties.line.dashStyle = "dash"

    chart.set_categories(dates)
    return chart


def create_backlog_chart(ws, min_row, max_row, date_col, backlog_col, title="Backlog Projection"):
    """Create an area chart for backlog evolution."""
    chart = AreaChart()
    chart.title = title
    chart.style = 10
    chart.y_axis.title = "Units"
    chart.x_axis.title = "Date"
    chart.width = 24
    chart.height = 14

    dates = Reference(ws, min_col=date_col, min_row=min_row + 1, max_row=max_row)
    data = Reference(ws, min_col=backlog_col, min_row=min_row, max_row=max_row)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(dates)

    series = chart.series[0]
    series.graphicalProperties.solidFill = "BDD7EE"
    series.graphicalProperties.line.solidFill = "2E75B6"

    return chart


def create_headcount_chart(ws, min_row, max_row, date_col, hc_cols, labels, title="Headcount Plan"):
    """Create a bar chart for headcount."""
    chart = BarChart()
    chart.type = "col"
    chart.title = title
    chart.style = 10
    chart.y_axis.title = "Workers"
    chart.x_axis.title = "Date"
    chart.width = 24
    chart.height = 14

    dates = Reference(ws, min_col=date_col, min_row=min_row + 1, max_row=max_row)

    colors = ["2E75B6", "ED7D31", "A5A5A5"]
    for i, col in enumerate(hc_cols):
        data = Reference(ws, min_col=col, min_row=min_row, max_row=max_row)
        chart.add_data(data, titles_from_data=True)
        if i < len(colors):
            chart.series[-1].graphicalProperties.solidFill = colors[i]

    chart.set_categories(dates)
    return chart


def create_accuracy_chart(ws, min_row, max_row, date_col, forecast_col, actual_col, title="Forecast vs Actual"):
    """Create an overlay line chart comparing forecast to actual."""
    chart = LineChart()
    chart.title = title
    chart.style = 10
    chart.y_axis.title = "Volume (units)"
    chart.x_axis.title = "Date"
    chart.width = 24
    chart.height = 14

    dates = Reference(ws, min_col=date_col, min_row=min_row + 1, max_row=max_row)

    for col, color in [(forecast_col, "2E75B6"), (actual_col, "ED7D31")]:
        data = Reference(ws, min_col=col, min_row=min_row, max_row=max_row)
        chart.add_data(data, titles_from_data=True)
        chart.series[-1].graphicalProperties.line.solidFill = color

    chart.set_categories(dates)
    return chart
