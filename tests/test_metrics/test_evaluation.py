import numpy as np
import pytest

from hireplanner.metrics.evaluation import evaluate_forecast, mae, mape, rmse, wape


class TestWape:
    def test_perfect_forecast(self):
        actual = np.array([100, 200, 300])
        predicted = np.array([100, 200, 300])
        assert wape(actual, predicted) == 0.0

    def test_known_values(self):
        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 280])
        # |100-110| + |200-190| + |300-280| = 10+10+20 = 40
        # sum(|actual|) = 600
        # wape = 40/600 = 0.06667
        assert wape(actual, predicted) == pytest.approx(40 / 600)

    def test_all_zeros_actual(self):
        actual = np.array([0, 0, 0])
        predicted = np.array([10, 20, 30])
        assert wape(actual, predicted) == 0.0

    def test_single_value(self):
        assert wape(np.array([200]), np.array([250])) == pytest.approx(50 / 200)

    def test_large_errors(self):
        actual = np.array([100, 100])
        predicted = np.array([300, 300])
        # errors = 200+200=400, total=200, wape=2.0
        assert wape(actual, predicted) == pytest.approx(2.0)


class TestMape:
    def test_perfect_forecast(self):
        actual = np.array([100, 200, 300])
        predicted = np.array([100, 200, 300])
        assert mape(actual, predicted) == 0.0

    def test_known_values(self):
        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 280])
        # |10/100| + |10/200| + |20/300| = 0.1 + 0.05 + 0.0667
        expected = (0.1 + 0.05 + 20 / 300) / 3
        assert mape(actual, predicted) == pytest.approx(expected)

    def test_all_zeros_actual(self):
        assert mape(np.array([0, 0, 0]), np.array([1, 2, 3])) == 0.0

    def test_skips_zeros_in_actual(self):
        actual = np.array([0, 100, 200])
        predicted = np.array([50, 110, 220])
        # Only indices 1,2 count: |10/100|=0.1, |20/200|=0.1 => mean=0.1
        assert mape(actual, predicted) == pytest.approx(0.1)


class TestMae:
    def test_perfect_forecast(self):
        assert mae(np.array([10, 20]), np.array([10, 20])) == 0.0

    def test_known_values(self):
        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 280])
        # mean(10,10,20) = 13.333
        assert mae(actual, predicted) == pytest.approx(40 / 3)


class TestRmse:
    def test_perfect_forecast(self):
        assert rmse(np.array([5, 10]), np.array([5, 10])) == 0.0

    def test_known_values(self):
        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 280])
        # errors squared: 100, 100, 400 => mean=200 => sqrt=14.142
        assert rmse(actual, predicted) == pytest.approx(np.sqrt(200))

    def test_single_value(self):
        assert rmse(np.array([50]), np.array([60])) == pytest.approx(10.0)


class TestEvaluateForecast:
    def test_returns_all_keys(self):
        result = evaluate_forecast([100, 200], [110, 190])
        assert set(result.keys()) == {"wape", "mape", "mae", "rmse"}

    def test_values_match_individual_functions(self):
        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 280])
        result = evaluate_forecast(actual, predicted)
        assert result["wape"] == pytest.approx(wape(actual, predicted))
        assert result["mape"] == pytest.approx(mape(actual, predicted))
        assert result["mae"] == pytest.approx(mae(actual, predicted))
        assert result["rmse"] == pytest.approx(rmse(actual, predicted))

    def test_accepts_lists(self):
        result = evaluate_forecast([100, 200], [100, 200])
        assert result["wape"] == 0.0
        assert result["mae"] == 0.0
