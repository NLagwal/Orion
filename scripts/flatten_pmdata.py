#!/usr/bin/env python3
"""
scripts/flatten_pmdata.py

Flattens the raw PMData dataset into a unified participant-day tabular dataset.
Joins across:
  - pmsys/wellness.csv (readiness, fatigue, soreness, stress, mood)
  - pmsys/srpe.csv (sRPE * duration -> daily training load, rolling 7d/28d average, ACWR)
  - fitbit/resting_heart_rate.json (resting HR, 7d rolling average, delta trend)
  - fitbit/sleep.json (efficiency, duration, sleep stages)
  - pmsys/injury.csv (sparse daily injury flags)
  - participant-overview.xlsx (static demographics, max HR, baseline 5km fitness)

Outputs:
  - data/pmdata_flattened.csv
  - data/pmdata_summary.json
"""

import os
import sys
import csv
import json
import zipfile
import datetime
import xml.etree.ElementTree as ET
from collections import defaultdict


def parse_participant_overview(xlsx_path):
    """
    Parses participant-overview.xlsx directly from XML to avoid external dependencies.
    Extracts age, gender, height, max HR, chronotype, and baseline 5km run time (in seconds).
    """
    if not os.path.exists(xlsx_path):
        print(f"[WARN] Overview file not found at {xlsx_path}", file=sys.stderr)
        return {}

    with zipfile.ZipFile(xlsx_path) as z:
        shared_strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for elem in tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                t = elem.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')
                shared_strings.append(t.text.strip() if (t is not None and t.text) else '')

        sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows = []
        for row in sheet.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row'):
            row_vals = []
            for c in row.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                t = c.attrib.get('t')
                v = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                val = v.text.strip() if (v is not None and v.text) else ''
                if t == 's' and val.isdigit():
                    idx = int(val)
                    val = shared_strings[idx] if idx < len(shared_strings) else ''
                row_vals.append(val.strip())
            if row_vals:
                rows.append(row_vals)

    data = {}
    for r in rows[2:]:
        if not r or not r[0].startswith('p'):
            continue
        pid = r[0].strip()
        age = int(r[1]) if (len(r) > 1 and r[1].isdigit()) else None
        height = int(r[2]) if (len(r) > 2 and r[2].isdigit()) else None
        gender = r[3].strip().lower() if len(r) > 3 else None
        chronotype = r[4].strip() if len(r) > 4 else None
        max_hr_raw = r[5].strip() if len(r) > 5 else ''
        max_hr = int(max_hr_raw) if max_hr_raw.isdigit() else (220 - age if age else None)

        mins = r[7].strip() if len(r) > 7 else ''
        secs = r[8].strip() if len(r) > 8 else ''
        if mins.isdigit() and secs.isdigit():
            b5k = int(mins) * 60 + int(secs)
        else:
            b5k = None

        data[pid] = {
            'age': age,
            'gender': gender,
            'height_cm': height,
            'max_hr': max_hr,
            'chronotype_group': chronotype,
            'baseline_5k_sec': b5k
        }
    return data


def parse_wellness(csv_path):
    """
    Parses pmsys/wellness.csv.
    If multiple entries exist for a single date, keeps the latest entry.
    """
    if not os.path.exists(csv_path):
        return {}

    wellness_by_date = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get('effective_time_frame', '')
            if not ts:
                continue
            date_str = ts[:10]
            existing = wellness_by_date.get(date_str)
            # If duplicate date, keep latest timestamp
            if existing is None or ts > existing['_ts']:
                wellness_by_date[date_str] = {
                    '_ts': ts,
                    'readiness': int(row['readiness']) if row.get('readiness', '').isdigit() else None,
                    'fatigue': int(row['fatigue']) if row.get('fatigue', '').isdigit() else None,
                    'soreness': int(row['soreness']) if row.get('soreness', '').isdigit() else None,
                    'stress': int(row['stress']) if row.get('stress', '').isdigit() else None,
                    'mood': int(row['mood']) if row.get('mood', '').isdigit() else None,
                    'sleep_quality': int(row['sleep_quality']) if row.get('sleep_quality', '').isdigit() else None,
                }
    return wellness_by_date


def parse_srpe(csv_path):
    """
    Parses pmsys/srpe.csv.
    Computes session_load = perceived_exertion * duration_min.
    Aggregates multiple sessions per day into daily_training_load and daily_workout_minutes.
    """
    if not os.path.exists(csv_path):
        return {}

    daily_srpe = defaultdict(lambda: {'daily_load': 0.0, 'daily_minutes': 0.0, 'session_count': 0})
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get('end_date_time', '')
            if not ts:
                continue
            date_str = ts[:10]
            try:
                rpe = float(row.get('perceived_exertion', 0) or 0)
                dur = float(row.get('duration_min', 0) or 0)
            except (ValueError, TypeError):
                continue
            load = rpe * dur
            daily_srpe[date_str]['daily_load'] += load
            daily_srpe[date_str]['daily_minutes'] += dur
            daily_srpe[date_str]['session_count'] += 1

    return dict(daily_srpe)


def parse_resting_hr(json_path):
    """
    Parses fitbit/resting_heart_rate.json.
    Keyed by date YYYY-MM-DD.
    """
    if not os.path.exists(json_path):
        return {}

    rhr_by_date = {}
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {}

        for item in data:
            dt = item.get('dateTime', '')
            if not dt:
                continue
            date_str = dt[:10]
            val = item.get('value', {}).get('value')
            if val is not None:
                try:
                    rhr_by_date[date_str] = round(float(val), 2)
                except (ValueError, TypeError):
                    pass
    return rhr_by_date


def parse_sleep(json_path):
    """
    Parses fitbit/sleep.json.
    Keyed by dateOfSleep. Prefers mainSleep == True; falls back to longest duration sleep.
    Extracts duration, efficiency, deep, rem, light, wake minutes.
    """
    if not os.path.exists(json_path):
        return {}

    sleep_by_date = {}
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {}

        for item in data:
            date_str = item.get('dateOfSleep', '')
            if not date_str:
                continue

            is_main = bool(item.get('mainSleep', False))
            duration = item.get('duration', 0) or 0
            existing = sleep_by_date.get(date_str)

            # Prioritize mainSleep, else take longest duration
            if existing is None or (is_main and not existing['_is_main']) or (is_main == existing['_is_main'] and duration > existing['_duration']):
                summary = item.get('levels', {}).get('summary', {})
                deep = summary.get('deep', {}).get('minutes')
                rem = summary.get('rem', {}).get('minutes')
                light = summary.get('light', {}).get('minutes')
                wake = summary.get('wake', {}).get('minutes')
                if wake is None and 'awake' in summary:
                    wake = summary.get('awake', {}).get('minutes')

                sleep_by_date[date_str] = {
                    '_is_main': is_main,
                    '_duration': duration,
                    'sleep_duration_min': item.get('minutesAsleep'),
                    'sleep_efficiency': item.get('efficiency'),
                    'sleep_deep_min': deep,
                    'sleep_rem_min': rem,
                    'sleep_light_min': light,
                    'sleep_wake_min': wake
                }

    return sleep_by_date


def parse_injuries(csv_path):
    """
    Parses pmsys/injury.csv.
    Keyed by date YYYY-MM-DD.
    Sets injury_reported = 1 if non-empty injury dict exists, else 0.
    """
    if not os.path.exists(csv_path):
        return {}

    injuries_by_date = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get('effective_time_frame', '')
            if not ts:
                continue
            date_str = ts[:10]
            raw_injuries = row.get('injuries', '').strip()
            if raw_injuries and raw_injuries != '{}':
                injuries_by_date[date_str] = {
                    'injury_reported': 1,
                    'injury_details': raw_injuries
                }
            else:
                if date_str not in injuries_by_date:
                    injuries_by_date[date_str] = {
                        'injury_reported': 0,
                        'injury_details': ''
                    }
    return injuries_by_date


def flatten_pmdata(root_dir, output_csv, output_summary_json):
    """
    Main orchestration function to extract, join, calculate rolling windows, and export.
    """
    overview_path = os.path.join(root_dir, 'participant-overview.xlsx')
    participants_meta = parse_participant_overview(overview_path)

    participant_dirs = [
        d for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('p')
    ]

    all_rows = []
    summary_stats = {
        'total_participants': len(participant_dirs),
        'participant_details': {},
        'total_rows': 0,
        'date_range': {}
    }

    global_min_date = None
    global_max_date = None

    for pid in participant_dirs:
        p_dir = os.path.join(root_dir, pid)
        meta = participants_meta.get(pid, {})

        wellness = parse_wellness(os.path.join(p_dir, 'pmsys', 'wellness.csv'))
        srpe = parse_srpe(os.path.join(p_dir, 'pmsys', 'srpe.csv'))
        rhr = parse_resting_hr(os.path.join(p_dir, 'fitbit', 'resting_heart_rate.json'))
        sleep = parse_sleep(os.path.join(p_dir, 'fitbit', 'sleep.json'))
        injuries = parse_injuries(os.path.join(p_dir, 'pmsys', 'injury.csv'))

        # Union of all dates where participant has any logged signal
        all_dates_set = set(wellness.keys()) | set(srpe.keys()) | set(rhr.keys()) | set(sleep.keys()) | set(injuries.keys())
        if not all_dates_set:
            continue

        min_date_str = min(all_dates_set)
        max_date_str = max(all_dates_set)

        if global_min_date is None or min_date_str < global_min_date:
            global_min_date = min_date_str
        if global_max_date is None or max_date_str > global_max_date:
            global_max_date = max_date_str

        min_date = datetime.date.fromisoformat(min_date_str)
        max_date = datetime.date.fromisoformat(max_date_str)

        total_days = (max_date - min_date).days + 1
        summary_stats['participant_details'][pid] = {
            'min_date': min_date_str,
            'max_date': max_date_str,
            'total_calendar_days': total_days,
            'wellness_entries': len(wellness),
            'workout_sessions': sum(s['session_count'] for s in srpe.values()),
            'rhr_entries': len(rhr),
            'sleep_entries': len(sleep),
            'injury_reports': sum(1 for inj in injuries.values() if inj['injury_reported'] == 1)
        }

        # Rolling buffers
        past_loads = []       # list of daily loads
        past_rhrs = []        # list of (date, rhr_value)

        curr_date = min_date
        while curr_date <= max_date:
            d_str = curr_date.isoformat()

            # 1. Training load & workout minutes (zero-padded on rest days)
            srpe_data = srpe.get(d_str, {'daily_load': 0.0, 'daily_minutes': 0.0})
            daily_load = srpe_data['daily_load']
            daily_mins = srpe_data['daily_minutes']

            past_loads.append(daily_load)

            # 7-day rolling average load (Acute)
            window_7 = past_loads[-7:]
            rolling_load_7d = round(sum(window_7) / len(window_7), 2)

            # 28-day rolling average load (Chronic)
            window_28 = past_loads[-28:]
            rolling_load_28d = round(sum(window_28) / len(window_28), 2)

            # ACWR
            if rolling_load_28d > 0:
                acwr = round(rolling_load_7d / rolling_load_28d, 3)
            else:
                acwr = None

            # 2. Resting Heart Rate & Trend
            curr_rhr = rhr.get(d_str)
            if curr_rhr is not None:
                past_rhrs.append(curr_rhr)
            else:
                past_rhrs.append(None)

            # 7-day rolling RHR
            rhr_window_7 = [x for x in past_rhrs[-7:] if x is not None]
            if rhr_window_7:
                rhr_rolling_7d = round(sum(rhr_window_7) / len(rhr_window_7), 2)
                if curr_rhr is not None:
                    rhr_trend_delta = round(curr_rhr - rhr_rolling_7d, 2)
                else:
                    rhr_trend_delta = None
            else:
                rhr_rolling_7d = None
                rhr_trend_delta = None

            # 3. Wellness metrics
            w_entry = wellness.get(d_str, {})
            # 4. Sleep metrics
            s_entry = sleep.get(d_str, {})
            # 5. Injury
            inj_entry = injuries.get(d_str, {'injury_reported': 0, 'injury_details': ''})

            row = {
                'participant_id': pid,
                'date': d_str,
                'age': meta.get('age', ''),
                'gender': meta.get('gender', ''),
                'height_cm': meta.get('height_cm', ''),
                'max_hr': meta.get('max_hr', ''),
                'chronotype_group': meta.get('chronotype_group', ''),
                'baseline_5k_sec': meta.get('baseline_5k_sec', ''),
                'readiness': w_entry.get('readiness', ''),
                'fatigue': w_entry.get('fatigue', ''),
                'soreness': w_entry.get('soreness', ''),
                'stress': w_entry.get('stress', ''),
                'mood': w_entry.get('mood', ''),
                'sleep_quality': w_entry.get('sleep_quality', ''),
                'daily_training_load': round(daily_load, 1),
                'daily_workout_minutes': round(daily_mins, 1),
                'rolling_load_7d': rolling_load_7d,
                'rolling_load_28d': rolling_load_28d,
                'acwr': acwr if acwr is not None else '',
                'resting_heart_rate': curr_rhr if curr_rhr is not None else '',
                'rhr_rolling_7d': rhr_rolling_7d if rhr_rolling_7d is not None else '',
                'rhr_trend_delta': rhr_trend_delta if rhr_trend_delta is not None else '',
                'sleep_duration_min': s_entry.get('sleep_duration_min', ''),
                'sleep_efficiency': s_entry.get('sleep_efficiency', ''),
                'sleep_deep_min': s_entry.get('sleep_deep_min', ''),
                'sleep_rem_min': s_entry.get('sleep_rem_min', ''),
                'sleep_light_min': s_entry.get('sleep_light_min', ''),
                'sleep_wake_min': s_entry.get('sleep_wake_min', ''),
                'injury_reported': inj_entry['injury_reported'],
                'injury_details': inj_entry['injury_details']
            }

            all_rows.append(row)
            curr_date += datetime.timedelta(days=1)

    summary_stats['total_rows'] = len(all_rows)
    summary_stats['date_range'] = {'min_date': global_min_date, 'max_date': global_max_date}

    # Ensure target output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fieldnames = [
        'participant_id', 'date', 'age', 'gender', 'height_cm', 'max_hr',
        'chronotype_group', 'baseline_5k_sec', 'readiness', 'fatigue',
        'soreness', 'stress', 'mood', 'sleep_quality', 'daily_training_load',
        'daily_workout_minutes', 'rolling_load_7d', 'rolling_load_28d', 'acwr',
        'resting_heart_rate', 'rhr_rolling_7d', 'rhr_trend_delta',
        'sleep_duration_min', 'sleep_efficiency', 'sleep_deep_min',
        'sleep_rem_min', 'sleep_light_min', 'sleep_wake_min',
        'injury_reported', 'injury_details'
    ]

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    with open(output_summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2)

    print(f"Successfully flattened PMData into {output_csv}")
    print(f"Total participants: {summary_stats['total_participants']}")
    print(f"Total rows: {summary_stats['total_rows']}")
    print(f"Date range: {global_min_date} to {global_max_date}")
    print(f"Summary written to {output_summary_json}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pmdata_dir = os.path.join(base_dir, 'osfstorage-archive', 'pmdata')
    out_csv = os.path.join(base_dir, 'data', 'pmdata_flattened.csv')
    out_json = os.path.join(base_dir, 'data', 'pmdata_summary.json')

    flatten_pmdata(pmdata_dir, out_csv, out_json)
