"""
tests/test_flattened_pmdata.py

Automated tests validating the integrity of the flattened PMData dataset.
"""

import os
import csv
import datetime


def test_flattened_dataset_integrity():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'pmdata_flattened.csv')
    assert os.path.exists(csv_path), f"File {csv_path} does not exist"

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # 1. Column Presence
    expected_cols = [
        'participant_id', 'date', 'age', 'gender', 'height_cm', 'max_hr',
        'chronotype_group', 'baseline_5k_sec', 'readiness', 'fatigue',
        'soreness', 'stress', 'mood', 'sleep_quality', 'daily_training_load',
        'daily_workout_minutes', 'rolling_load_7d', 'rolling_load_28d', 'acwr',
        'resting_heart_rate', 'rhr_rolling_7d', 'rhr_trend_delta',
        'sleep_duration_min', 'sleep_efficiency', 'sleep_deep_min',
        'sleep_rem_min', 'sleep_light_min', 'sleep_wake_min',
        'injury_reported', 'injury_details'
    ]
    for col in expected_cols:
        assert col in fieldnames, f"Missing expected column: {col}"

    # 2. Participant Coverage
    pids = sorted(list({r['participant_id'] for r in rows}))
    expected_pids = [f'p{i:02d}' for i in range(1, 17)]
    assert pids == expected_pids, f"Expected 16 participants, got {len(pids)}: {pids}"

    # 3. Row count sanity
    assert len(rows) == 2354, f"Expected 2354 rows, got {len(rows)}"

    # 4. Date continuity per participant
    rows_by_pid = {}
    for r in rows:
        rows_by_pid.setdefault(r['participant_id'], []).append(r)

    for pid, p_rows in rows_by_pid.items():
        dates = [datetime.date.fromisoformat(r['date']) for r in p_rows]
        for i in range(len(dates) - 1):
            diff = (dates[i+1] - dates[i]).days
            assert diff == 1, f"Gap detected for {pid} between {dates[i]} and {dates[i+1]}"

    # 5. Calculation Sanity
    for r in rows:
        # Check resting HR trend delta consistency
        if r['resting_heart_rate'] and r['rhr_rolling_7d'] and r['rhr_trend_delta']:
            rhr = float(r['resting_heart_rate'])
            rhr_7d = float(r['rhr_rolling_7d'])
            delta = float(r['rhr_trend_delta'])
            assert abs((rhr - rhr_7d) - delta) < 1e-2, f"RHR delta mismatch on {r['participant_id']} {r['date']}"

        # Check injury flag consistency
        if r['injury_reported'] == '1':
            assert len(r['injury_details']) > 0, f"Injury reported but details empty on {r['participant_id']} {r['date']}"
        elif r['injury_reported'] == '0':
            assert r['injury_details'] == '', f"Injury reported is 0 but details present on {r['participant_id']} {r['date']}"

    print("All automated verification tests passed successfully!")


if __name__ == '__main__':
    test_flattened_dataset_integrity()
