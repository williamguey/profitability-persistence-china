"""Read-only recovery of supplied uncertain thesis files; no workbook recalculation.

Run with bundled Python (openpyxl). Preserves original files and both stored
formula strings and cached scalar values in separate CSVs. CSV empty fields
represent empty cells; cell-types JSON preserves errors/formula distinctions.
"""
from pathlib import Path
import argparse, csv, hashlib, json, re, shutil
from collections import Counter
import openpyxl

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workbook', type=Path, help='Path to the supplied workbook. Default: INPUT_DIR/Dados Tese.xlsx.')
parser.add_argument('--input-dir', type=Path, help='Directory containing the workbook and optional Tese*.R files. Default: data/original-private, or the explicit workbook directory.')
parser.add_argument('--output-dir', type=Path, default=ROOT/'data/original-private', help='Extraction destination; defaults to project data/original-private.')
args = parser.parse_args()
OUT = args.output_dir.expanduser().resolve()
SRC = (args.input_dir or (args.workbook.parent if args.workbook else ROOT/'data/original-private')).expanduser().resolve()
workbook = (args.workbook or SRC/'Dados Tese.xlsx').expanduser().resolve()
if not workbook.is_file():
    parser.error(f'Workbook not found: {workbook}. Supply --workbook PATH or place Dados Tese.xlsx in data/original-private.')
OUT.mkdir(parents=True, exist_ok=True)
sources = [(workbook, 'Dados Tese.xlsx')]
sources += [(SRC/name,name) for name in ['Tese.R','Tese(1).R','Tese(2).R'] if (SRC/name).is_file()]
manifest = {'status': 'User-supplied uncertain versions; not assumed final', 'files': [], 'sheets': []}
for source, name in sources:
    raw = source.read_bytes()
    dst = OUT/name
    if dst.exists():
        assert dst.read_bytes() == raw, f'Existing different source: {dst}'
    else:
        shutil.copy2(source, dst)
    manifest['files'].append({'supplied_path': str(source), 'preserved_path': str(dst), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})

cached = openpyxl.load_workbook(OUT/'Dados Tese.xlsx', read_only=True, data_only=True)
formula = openpyxl.load_workbook(OUT/'Dados Tese.xlsx', read_only=True, data_only=False)
assert cached.worksheets[0].title == 'Base Tese', 'Expected the original Base Tese as the first worksheet.'
for i, (ws, wf) in enumerate(zip(cached, formula), 1):
    slug = f'{i:02d}_' + re.sub(r'[^A-Za-z0-9]+','_', ws.title).strip('_').lower()
    counts = Counter()
    special = []
    row_count = 0
    with (OUT/(slug+'_cached.csv')).open('w',newline='',encoding='utf8') as a, (OUT/(slug+'_formulas.csv')).open('w',newline='',encoding='utf8') as b:
        wa,wb = csv.writer(a),csv.writer(b)
        for rc,rf in zip(ws.iter_rows(),wf.iter_rows()):
            row_count += 1
            wa.writerow([c.value for c in rc]); wb.writerow([c.value for c in rf])
            for cv,cf in zip(rc,rf):
                counts[cf.data_type] += 1
                if cf.data_type in ('f','e'):
                    special.append({'cell':cf.coordinate,'type':cf.data_type,'formula_or_error':cf.value,'cached_value':cv.value,'cached_type':cv.data_type})
    (OUT/(slug+'_special_cells.json')).write_text(json.dumps(special,ensure_ascii=False,indent=2,default=str))
    manifest['sheets'].append({'name':ws.title,'rows_including_header':row_count,'columns':ws.max_column,'cell_types':dict(counts),'prefix':slug})
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(manifest,ensure_ascii=False,indent=2))
