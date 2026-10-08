"""H2 Excel boundary: select timestamps before decoding measurement values.

Metadata inspection reads headers only. TRAIN loading never interpolates,
normalizes, selects a population or invents an observation/price interpretation.
XML bytes and timestamp metadata are traversed to locate the selected rows;
out-of-window measurement cells are not decoded into observations.
"""

from contextlib import contextmanager
from datetime import datetime, time
from functools import lru_cache
import hashlib
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from openpyxl.utils.datetime import CALENDAR_MAC_1904, CALENDAR_WINDOWS_1900, from_excel

from src.params.bm_1_history import BM1HistoricalCurveContract


_NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
_RID = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
INPUT_SHEET = '2022 med dato (3)'


class _Workbook:
    """Read selected XLSX cells without constructing whole-sheet DataFrames."""

    def __init__(self, path):
        self.path = Path(path)
        self.archive = zipfile.ZipFile(self.path)
        try:
            book = ET.fromstring(self.archive.read('xl/workbook.xml'))
            relationships = ET.fromstring(self.archive.read('xl/_rels/workbook.xml.rels'))
            targets = {r.attrib['Id']: r.attrib['Target'] for r in relationships}
            self.sheets = {}
            for sheet in book.find(_NS+'sheets'):
                target = targets[sheet.attrib[_RID]]
                member = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
                self.sheets[sheet.attrib['name']] = member
            properties = book.find(_NS+'workbookPr')
            mac = properties is not None and properties.get('date1904') in ('1', 'true')
            self.epoch = CALENDAR_MAC_1904 if mac else CALENDAR_WINDOWS_1900
            self.strings = {}
        except BaseException:
            self.archive.close()
            raise

    def close(self):
        self.archive.close()

    def rows(self, sheet):
        if sheet not in self.sheets:
            raise ValueError(f'Worksheet {sheet!r} is missing.')
        with self.archive.open(self.sheets[sheet]) as handle:
            for _, row in ET.iterparse(handle, events=('end',)):
                if row.tag != _NS+'row':
                    continue
                cells = {}
                for cell in row.findall(_NS+'c'):
                    letters = ''.join(c for c in cell.attrib['r'] if c.isalpha())
                    column = 0
                    for letter in letters:
                        column = column*26+ord(letter.upper())-64
                    cells[column] = cell
                yield int(row.attrib['r']), cells
                row.clear()

    def _string(self, index):
        if index not in self.strings:
            with self.archive.open('xl/sharedStrings.xml') as handle:
                count = 0
                for _, item in ET.iterparse(handle, events=('end',)):
                    if item.tag != _NS+'si':
                        continue
                    if count == index:
                        self.strings[index] = ''.join(t.text or '' for t in item.iter(_NS+'t'))
                        return self.strings[index]
                    count += 1
                    item.clear()
            raise ValueError('Invalid shared-string index.')
        return self.strings[index]

    def cell_value(self, cell):
        if cell is None:
            return None
        if cell.find(_NS+'f') is not None:
            raise ValueError('Selected cells must be raw values, not cached Excel formulas.')
        kind = cell.get('t', 'n')
        value = cell.find(_NS+'v')
        text = None if value is None else value.text
        if kind == 'inlineStr':
            return ''.join(t.text or '' for t in cell.iter(_NS+'t'))
        if kind == 's':
            return self._string(int(text))
        if kind == 'e':
            raise ValueError('Selected cell contains an Excel error.')
        if kind == 'b':
            raise ValueError('Boolean values are not accepted as timestamps or measurements.')
        if kind == 'd':
            return datetime.fromisoformat(text)
        if kind in ('str',):
            return text
        if text is None:
            return None
        number = float(text)
        return int(number) if np.isfinite(number) and number.is_integer() else number

    def headers(self, sheet, depth):
        rows = []
        iterator = self.rows(sheet)
        try:
            for number, cells in iterator:
                if number > depth:
                    break
                while len(rows) < number:
                    rows.append({})
                rows[number-1] = {c: self.cell_value(v) for c, v in cells.items()}
        finally:
            iterator.close()
        if len(rows) != depth:
            raise ValueError('Workbook does not contain the declared header rows.')
        if depth == 1:
            columns = {c: str(v).strip() for c, v in rows[0].items() if v is not None}
        else:
            # Match the legacy two-row header, including merged meter headings.
            columns, previous = {}, None
            for column in sorted(set(rows[0]) | set(rows[1])):
                top, bottom = rows[0].get(column), rows[1].get(column)
                if top is not None:
                    previous = top
                top = previous
                if bottom is None:
                    raise ValueError('The selected sheet has an incomplete two-row header.')
                columns[column] = str(bottom).strip() if top is None else f'{str(top).strip()}_{str(bottom).strip()}'
        names = list(columns.values())
        if not names or any(not n for n in names) or len(set(names)) != len(names):
            raise ValueError('Workbook headers must have nonempty, unique flattened names.')
        return columns


@contextmanager
def _workbook(path):
    book = _Workbook(path)
    try:
        yield book
    finally:
        book.close()


def _fingerprint(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def inspect_history_workbooks(input_path, price_path, *, input_sheet=INPUT_SHEET,
                              price_sheet=None):
    """Return header evidence and unresolved decisions; read no data rows."""
    with _workbook(input_path) as power, _workbook(price_path) as price:
        selected_price_sheet = next(iter(price.sheets)) if price_sheet is None else price_sheet
        power_columns = power.headers(input_sheet, 2)
        price_columns = price.headers(selected_price_sheet, 1)
        measurements = [c for c in power_columns.values() if c not in ('År', 'Måned', 'Dato', 'Time')]
        return dict(schema='bm1-h2-header-inspection-v1', completed=True,
            historical_fit_ready=False, observation_values_read=0,
            input=dict(filename=Path(input_path).name, sha256=_fingerprint(input_path),
                sheets=list(power.sheets), selected_sheet=input_sheet,
                headers=list(power_columns.values()), measurement_columns=measurements),
            prices=dict(filename=Path(price_path).name, sha256=_fingerprint(price_path),
                sheets=list(price.sheets), selected_sheet=selected_price_sheet,
                headers=list(price_columns.values())),
            required_decisions=[
                'Document a fixed population and its measurement-column membership.',
                'Establish physical units and point versus interval observation meaning.',
                'Establish the demand timestamp timezone and DST/interval labeling.',
                'Declare consumer tariff or explicitly limited spot-price proxy and area.',
                'Document TRAIN / MODEL_SELECTION / FINAL_HOLDOUT and prior exposure.',
                'Freeze physical/price scales and history-only baseline provenance.'],
            interpretation='Header evidence only; no units, tariff, blind split or fitting acceptance inferred.')


def _bounds(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError('Declare a nonempty timezone-aware half-open window.')
    hour_ns = pd.Timedelta(hours=1).value
    if start.value % hour_ns or end.value % hour_ns:
        raise ValueError('Selected window boundaries must align with an hourly grid.')
    expected = pd.date_range(start, end, freq='h', inclusive='left')
    if expected.empty or end-expected[-1] != pd.Timedelta(hours=1):
        raise ValueError('Selected window boundaries must align with an hourly grid.')
    return start, end, expected


def _numeric(book, cell):
    value = book.cell_value(cell)
    if value is None or isinstance(value, str) and not value.strip():
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('Selected measurement is not a number or an explicit missing value.') from exc
    if np.isinf(number):
        raise ValueError('Selected measurement must be finite or explicitly missing.')
    return number


def _date(value, book):
    if isinstance(value, (int, float)):
        value = from_excel(value, epoch=book.epoch)
    if isinstance(value, str):
        value = pd.to_datetime(value, dayfirst=True, errors='raise')
    return pd.Timestamp(value).normalize()


def _has_raw_content(cells):
    # Ignore formatting-only rows without decoding their measurement cells.
    return any(cell.find(_NS+'f') is not None or
               any(node.text for node in cell.iter() if node.tag in (_NS+'v', _NS+'t'))
               for cell in cells.values())


def read_raw_power_window(path, *, columns, start, end, timezone,
                          sheet=INPUT_SHEET):
    """Decode only a declared fixed population inside [start,end); retain NaNs.

    Demand timezone is a required declaration. Ambiguous/nonexistent local
    timestamps fail rather than averaging, guessing a fold or shifting an hour.
    Missing hourly rows are inserted as missing, never imputed.
    """
    start, end, expected = _bounds(start, end)
    columns = tuple(columns) if not isinstance(columns, str) else ()
    if not columns or len(set(columns)) != len(columns):
        raise ValueError('Declare unique measurement columns explicitly.')
    if not isinstance(timezone, str) or not timezone.strip():
        raise ValueError('Declare demand timestamp timezone.')
    local_start, local_end = (t.tz_convert(timezone).tz_localize(None) for t in (start, end))
    from src.tools.import_data.import_input import input_data

    @lru_cache(maxsize=128, typed=True)
    def metadata_time_delta(raw_time):
        # Clock labels repeat across dates. Cache only this pure conversion,
        # per read, retaining every row's timestamp and selection checks below.
        return input_data._time_delta(pd.Series([raw_time])).iloc[0]

    times, readings = [], []
    with _workbook(path) as book:
        header = book.headers(sheet, 2)
        lookup = {v: k for k, v in header.items()}
        if any(c not in lookup for c in ('Dato', 'Time', *columns)):
            raise ValueError('Selected sheet lacks required timestamp/population columns.')
        if any(c in ('År', 'Måned', 'Dato', 'Time') for c in columns):
            raise ValueError('Metadata columns cannot be measurement population members.')
        previous_date = None
        for row_number, cells in book.rows(sheet):
            if row_number <= 2:
                continue
            raw_date = book.cell_value(cells.get(lookup['Dato']))
            raw_time = book.cell_value(cells.get(lookup['Time']))
            if raw_date is None and raw_time is None:
                if _has_raw_content(cells):
                    raise ValueError(f'Row {row_number} has cells but no timestamp metadata.')
                continue
            if raw_date is not None:
                previous_date = _date(raw_date, book)
            if previous_date is None or pd.isna(previous_date) or raw_time is None:
                raise ValueError(f'Row {row_number} has incomplete timestamp metadata.')
            if isinstance(raw_time, time):
                delta = pd.Timedelta(hours=raw_time.hour, minutes=raw_time.minute,
                                     seconds=raw_time.second, microseconds=raw_time.microsecond)
            elif isinstance(raw_time, (int, float)) and 0 <= raw_time < 1:
                # Excel stores fractions of a day as binary floats. Resolve
                # them at microsecond precision, not spurious nanoseconds.
                delta = pd.Timedelta(microseconds=round(raw_time*86400*1e6))
            else:
                delta = metadata_time_delta(raw_time)
            naive = previous_date+delta
            if pd.isna(naive):
                raise ValueError(f'Row {row_number} has invalid timestamp metadata.')
            if not local_start <= naive < local_end:
                continue  # No measurement cells decoded before this selection.
            stamp = naive.tz_localize(timezone, ambiguous='raise', nonexistent='raise').tz_convert(start.tz)
            if start <= stamp < end:
                times.append(stamp)
                readings.append([_numeric(book, cells.get(lookup[c])) for c in columns])
    index = pd.DatetimeIndex(times, tz=start.tz)
    if index.has_duplicates or not index.isin(expected).all():
        raise ValueError('Selected demand has duplicate or off-grid timestamps.')
    return pd.DataFrame(readings, index=index, columns=columns, dtype=float).reindex(expected)


def read_raw_spot_window(path, *, start, end, price_area, sheet=None):
    """Read raw SpotPriceDKK in a selected UTC window; no tariff inference/fill."""
    start, end, expected = _bounds(start, end)
    if price_area not in ('DK1', 'DK2'):
        raise ValueError('Explicitly select DK1 or DK2.')
    times, readings = [], []
    with _workbook(path) as book:
        sheet = next(iter(book.sheets)) if sheet is None else sheet
        lookup = {v: k for k, v in book.headers(sheet, 1).items()}
        if any(c not in lookup for c in ('HourUTC', 'PriceArea', 'SpotPriceDKK')):
            raise ValueError('Price sheet lacks required UTC/area/spot columns.')
        for row_number, cells in book.rows(sheet):
            if row_number <= 1:
                continue
            value = book.cell_value(cells.get(lookup['HourUTC']))
            if value is None:
                if _has_raw_content(cells):
                    raise ValueError(f'Price row {row_number} has no UTC timestamp.')
                continue
            if isinstance(value, (int, float)):
                value = from_excel(value, epoch=book.epoch)
            stamp = pd.Timestamp(value)
            stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
            if pd.isna(stamp):
                raise ValueError(f'Price row {row_number} has invalid UTC timestamp.')
            if not start <= stamp < end:
                continue
            area = str(book.cell_value(cells.get(lookup['PriceArea']))).strip().upper()
            if area != price_area:
                continue
            times.append(stamp.tz_convert(start.tz))
            readings.append(_numeric(book, cells.get(lookup['SpotPriceDKK'])))
    index = pd.DatetimeIndex(times, tz=start.tz)
    if index.has_duplicates or not index.isin(expected).all():
        raise ValueError('Selected spot prices have duplicate or off-grid timestamps.')
    return pd.Series(readings, index=index, dtype=float, name='SpotPriceDKK').reindex(expected)


def load_historical_training_workbooks(input_path, price_path, *, contract,
                                      origin, timezone, price_area):
    """Raw TRAIN boundary for the existing H adapter; caller supplies baseline.

    This spot-file route supports only an explicit spot_proxy contract. A real
    consumer_price requires an independently documented tariff input adapter.
    Metadata declarations are prerequisites, not independent proof of units.
    """
    if not isinstance(contract, BM1HistoricalCurveContract):
        raise TypeError('Declare the historical curve contract before loading TRAIN.')
    if contract.price_role != 'spot_proxy' or contract.price_unit != 'DKK/MWh':
        raise ValueError('SpotPriceDKK route requires declared spot_proxy in DKK/MWh; consumer tariff needs its own input.')
    origin = pd.Timestamp(origin)
    if pd.isna(origin) or origin.tzinfo is None or origin >= contract.split.train_start:
        raise ValueError('Declare a prior origin strictly before TRAIN.')
    origin = origin.tz_convert(contract.split.train_start.tz)
    input_digest, price_digest = _fingerprint(input_path), _fingerprint(price_path)
    power = read_raw_power_window(input_path, columns=contract.measurement_columns,
        start=contract.split.train_start, end=contract.split.selection_start, timezone=timezone)
    prices = read_raw_spot_window(price_path, start=origin, end=contract.split.selection_start,
        price_area=price_area)
    if input_digest != _fingerprint(input_path) or price_digest != _fingerprint(price_path):
        raise ValueError('Raw workbook changed while selecting TRAIN; retry on unchanged inputs.')
    metadata = dict(schema='bm1-h2-raw-training-inputs-v1', contract=contract.to_record(),
        origin=origin.isoformat(), demand_timestamp_timezone=timezone, price_area=price_area,
        input_sha256=input_digest, prices_sha256=price_digest,
        missing_policy='explicit hourly NaN; no interpolation, normalization or clipping',
        observation_boundary='only selected TRAIN measurements decoded; prices also cover explicit prior origin')
    return power, prices, metadata
