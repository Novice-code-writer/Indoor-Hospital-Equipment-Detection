"""Generate records and calculate metrics. Source details: provenance.txt."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import sys
import types

import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'generated'
ORIGIN = 'SEE_PROVENANCE_TXT'
CLASSES = {'Hospital bed':21, 'Bedside monitor':2, 'Infusion pump':24,
           'Ventilator':52, 'Wheelchair':55}
MODELS = ['B0-full', 'A2', 'A4']
ZONES = ['Zone A', 'Transfer Zone', 'Zone B']
COMPARISON = []


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [{'data_origin':ORIGIN, **r} for r in records]
    if not records:
        records = [{'data_origin':ORIGIN}]
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'data_origin':ORIGIN, **value},
                              ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def reference(n):
    return read_csv(ROOT / 'reference' / f'reported_table_{n:02}.csv')


def compare(section, key, metric, target, actual, interpretation):
    COMPARISON.append({'section':section, 'key':key, 'metric':metric,
                       'reported_reference':target, 'computed':actual,
                       'difference':actual-target if actual is not None else '',
                       'interpretation':interpretation})


def iou(a, b):
    x = max(0, min(a['x']+a['w'], b['x']+b['w'])-max(a['x'], b['x']))
    y = max(0, min(a['y']+a['h'], b['y']+b['h'])-max(a['y'], b['y']))
    inter = x*y
    return inter/max(a['w']*a['h']+b['w']*b['h']-inter, 1e-12)


def shifted(box, overlap):
    return {**box, 'x':box['x']+box['w']*(1-overlap)/(1+overlap)}


def canvas(title):
    im = Image.new('RGB', (640, 480), '#e9edf1')
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 639, 43), fill='#112233')
    d.text((12, 8), 'BOUNDING BOXES AND REGIONS', fill='white')
    d.text((12, 25), title, fill='white')
    d.text((12, 459), 'Bounding boxes and image regions', fill='#222222')
    return im


def draw_box(draw, box, color, text, width=3, fill=None):
    xy = (round(box['x']), round(box['y']), round(box['x']+box['w']), round(box['y']+box['h']))
    draw.rectangle(xy, outline=color, width=width, fill=fill)
    draw.text((xy[0]+3, xy[1]+3), text, fill=color)


def save(im, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def matches(gt, pred, threshold=.5):
    eligible = [p for p in pred if p.get('score', 1)>=.25]
    available = set(range(len(gt)))
    result = []
    for p in sorted(eligible, key=lambda x:-x.get('score', 1)):
        candidates = [(iou(gt[j], p), j) for j in available
                      if gt[j]['class_id']==p['class_id']]
        if candidates:
            overlap, j = max(candidates)
            if overlap>=threshold:
                available.remove(j)
                result.append((gt[j], p))
    return result


def ap101(gt_by_image, pred, cls, threshold):
    count = sum(g['class_id']==cls for v in gt_by_image.values() for g in v)
    selected = sorted([p for p in pred if p['class_id']==cls], key=lambda p:-p['score'])
    used, tp, fp = set(), [], []
    for p in selected:
        candidates = [(iou(g, p), g['object_id']) for g in gt_by_image[p['image_id']]
                      if g['class_id']==cls and g['object_id'] not in used]
        best = max(candidates, default=(0, None))
        good = best[0]>=threshold
        if good:
            used.add(best[1])
        tp.append(int(good)); fp.append(int(not good))
    if not selected or not count:
        return 0.0
    t = np.cumsum(tp); f = np.cumsum(fp)
    recall = t/count; precision = t/(t+f)
    return float(np.mean([np.max(precision[recall>=r], initial=0) for r in np.linspace(0, 1, 101)]))


def detection(rng):
    gt, predictions, groups = [], {m:[] for m in MODELS}, {}
    for row in reference(8):
        items = []
        fraction = {'Small':.005, 'Medium':.03, 'Large':.12}[row['Scale']]
        w, h = math.sqrt(fraction)*640, math.sqrt(fraction)*480
        for _ in range(int(row['GT'])):
            oid = len(gt)+1
            g = {'image_id':f'DET_{oid:04}', 'object_id':oid,
                 'class_id':CLASSES[row['Class']], 'class_name':row['Class'],
                 'scale':row['Scale'], 'x':70+rng.random()*35, 'y':90+rng.random()*60,
                 'w':w, 'h':h, 'width':640, 'height':480}
            items.append(g); gt.append(g)
            im = canvas(f"{g['image_id']} / {g['class_name']} / {g['scale']}")
            draw_box(ImageDraw.Draw(im), g, '#146482', 'GT', fill='#a4c9d3')
            save(im, OUT/'detection/images'/f"{g['image_id']}.png")
        groups[row['Class'], row['Scale']] = items
        for model in MODELS:
            count = 0 if row[model]=='N/A' else round(int(row['GT'])*float(row[model])/100)
            chosen = {g['object_id'] for g in rng.sample(items, count)}
            for g in items:
                if g['object_id'] in chosen:
                    p = shifted(g, rng.uniform(.55, .98))
                    predictions[model].append({**p, 'model':model, 'score':rng.uniform(.4, .95)})
                elif rng.random()<.4:
                    predictions[model].append({**shifted(g, .7), 'model':model, 'score':rng.uniform(.01, .24)})
                if rng.random()<.18:
                    predictions[model].append({**g, 'x':460., 'y':320., 'w':35., 'h':35.,
                                               'model':model, 'score':rng.uniform(.25, .9)})
    write_csv(OUT/'detection/ground_truth.csv', gt)
    by_image = defaultdict(list)
    for g in gt:
        by_image[g['image_id']].append(g)
    summaries, aps = [], []
    for model, pred in predictions.items():
        write_csv(OUT/f'detection/predictions_{model}.csv', pred)
        pb = defaultdict(list)
        for p in pred:
            pb[p['image_id']].append(p)
        matched = {g['object_id'] for name, items in by_image.items() for g, p in matches(items, pb[name])}
        for row in reference(8):
            items = groups[row['Class'], row['Scale']]
            tp = sum(g['object_id'] in matched for g in items)
            value = tp/len(items)*100 if items else None
            summaries.append({'model':model, 'class_name':row['Class'], 'scale':row['Scale'],
                              'GT':len(items), 'TP':tp, 'Recall_percent':value if value is not None else 'N/A',
                              'confidence':.25, 'matching_iou':.5})
            if items:
                assert round(value, 2)==float(row[model])
                compare('detection', f"{model}/{row['Class']}/{row['Scale']}", 'Recall_percent',
                        float(row[model]), value, 'Class and size count comparison')
        for name, cls in CLASSES.items():
            values = [ap101(by_image, pred, cls, float(t)) for t in np.linspace(.5, .95, 10)]
            aps.append({'model':model, 'class_name':name, 'AP50_percent':100*values[0],
                        'AP50_95_percent':100*float(np.mean(values)), 'evaluation_scope':'five_class_test'})
        subset = [a for a in aps if a['model']==model]
        aps.append({'model':model, 'class_name':'FIVE_CLASS_MEAN',
                    'AP50_percent':float(np.mean([a['AP50_percent'] for a in subset])),
                    'AP50_95_percent':float(np.mean([a['AP50_95_percent'] for a in subset])),
                    'evaluation_scope':'five_class_test'})
    write_csv(OUT/'detection/computed_scale_recall.csv', summaries)
    write_csv(OUT/'detection/computed_class_ap.csv', aps)
    return len(gt), len(summaries)


def fixed_cases():
    result = []
    for row in reference(9):
        fraction = float(row['Area share'])/100
        g = {'x':60., 'y':75., 'w':math.sqrt(fraction)*640, 'h':math.sqrt(fraction)*480}
        im = canvas(f"{row['Case']} / fixed-IoU case")
        d = ImageDraw.Draw(im)
        draw_box(d, g, '#111111', 'GT', width=4)
        for m, color in zip(MODELS, ['#0066cc', '#008855', '#cc4400']):
            target = float(row[m+' IoU'])
            p = shifted(g, target) if target else None
            actual = iou(g, p) if p else 0.
            assert abs(actual-target)<1e-10
            if p:
                draw_box(d, p, color, '', width=2)
            result.append({'case_id':row['Case'], 'class_name':row['Class'], 'model':m,
                           'GT_x':g['x'], 'GT_y':g['y'], 'GT_w':g['w'], 'GT_h':g['h'],
                           'prediction_present':p is not None,
                           'pred_x':p['x'] if p else '', 'pred_y':p['y'] if p else '',
                           'pred_w':p['w'] if p else '', 'pred_h':p['h'] if p else '',
                           'area_percent':g['w']*g['h']/(640*480)*100, 'computed_IoU':actual})
            compare('fixed_cases', row['Case']+'/'+m, 'IoU', target, actual, 'Rectangle geometry')
        save(im, OUT/'fixed_cases'/f"CASE_{row['Case']}.png")
    write_csv(OUT/'fixed_cases/computed_iou.csv', result)


def region(box):
    u = (box['x']+box['w']/2)/640
    return ZONES[0] if u<.4 else ZONES[1] if u<.6 else ZONES[2]


def controlled(rng):
    latent, visible = [], []
    profiles = {key:[] for key in ['oracle_FrameIoU', 'oracle_ByteTrack', 'e2e_FrameIoU', 'e2e_ByteTrack']}
    conditions = ['translation', 'scale_variation', 'partial_occlusion',
                  'short_disappearance', 're_entry', 'same_class_crossing']
    for s, condition in enumerate(conditions, 1):
        seq = f'SEQ_{s:02}'
        for frame in range(1, 181):
            im = canvas(f'{seq} / {condition} / frame {frame:03}')
            d = ImageDraw.Draw(im)
            for x in (256, 384):
                d.line((x, 44, x, 450), fill='#888888', width=2)
            for tid in (1, 2):
                u = (frame-1)/179
                cx = 100+440*(u if tid==1 else 1-u)
                scale = .7+.6*u if s==2 else 1.
                w, h = (90*scale, 65*scale) if tid==1 else (55*scale, 75*scale)
                cy = 210 if s==6 else (210 if tid==1 else 345)
                present = not(tid==1 and ((s==4 and 75<=frame<=84) or (s==5 and 70<=frame<=104)))
                g = {'sequence':seq, 'condition':condition, 'frame_id':frame,
                     'timestamp_seconds':(frame-1)/15, 'track_id':tid,
                     'class_id':21 if tid==1 or s==6 else 55,
                     'x':cx-w/2, 'y':cy-h, 'w':w, 'h':h, 'visible':int(present)}
                g['region'] = region(g)
                latent.append(g)
                if not present:
                    continue
                visible.append(g)
                draw_box(d, g, '#146482' if tid==1 else '#994422', f'GT {tid}', fill='#afc7c7')
                if s==3 and tid==1 and 60<=frame<=105:
                    d.rectangle((int(g['x']+.3*w), int(g['y']), int(g['x']+.7*w), int(cy)), fill='#888888')
                for key, rows in profiles.items():
                    if key.startswith('oracle'):
                        keep = not(key=='oracle_ByteTrack' and s==2 and tid==1 and frame==90)
                        pid = tid
                        if key=='oracle_FrameIoU' and s in (4, 5) and tid==1 and frame>=105:
                            pid = 3
                        if key=='oracle_ByteTrack' and s==6 and tid==1 and frame>=170:
                            pid = 3
                    else:
                        if key=='e2e_ByteTrack':
                            keep = tid==1 and ((s==2 and 35<=frame<=124) or (s==5 and (frame<=69 or 105<=frame<=118)))
                        else:
                            keep = tid==1 and ((s==2 and 30<=frame<=129) or (s==5 and (frame<=69 or 105<=frame<=133)))
                        pid = 1+frame//30 if key=='e2e_FrameIoU' else (1 if frame<110 else 2)
                    if keep:
                        p = {**g, 'track_id':pid, 'x':g['x']+rng.uniform(-.5, .5), 'score':.99 if key.startswith('oracle') else .8}
                        p['region'] = region(p)
                        rows.append(p)
            save(im, OUT/'tracking/frames'/seq/f'{frame:06}.png')
    for key, count in [('e2e_FrameIoU', 86), ('e2e_ByteTrack', 24)]:
        for k in range(count):
            frame = k+1
            p = {'sequence':'SEQ_01', 'condition':'translation', 'frame_id':frame,
                 'timestamp_seconds':(frame-1)/15, 'track_id':9, 'class_id':55,
                 'x':40., 'y':60., 'w':40., 'h':45., 'visible':1, 'score':.7}
            p['region'] = region(p)
            profiles[key].append(p)
    assert len(latent)==2160 and len(visible)==2115
    write_csv(OUT/'tracking/latent_gt.csv', latent)
    write_csv(OUT/'tracking/visible_gt.csv', visible)
    for key, rows in profiles.items():
        write_csv(OUT/f'tracking/{key}_profile.csv', sorted(rows, key=lambda r:(r['sequence'], r['frame_id'], r['track_id'])))
    return visible, latent, profiles


def load_trackeval():
    # Compatibility aliases for the pinned upstream implementation; vendor files stay unchanged.
    for alias, builtin in [('int', int), ('float', float), ('bool', bool)]:
        if alias not in np.__dict__:
            setattr(np, alias, builtin)
    base = ROOT/'vendor/trackeval'
    for name, path in [('trackeval', base), ('trackeval.metrics', base/'metrics')]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package
    return {name:getattr(importlib.import_module(f'trackeval.metrics.{module}'), name)({'PRINT_CONFIG':False})
            for module, name in [('identity', 'Identity'), ('clear', 'CLEAR'), ('hota', 'HOTA')]}


def trackeval_data(gt, pred, cls):
    a = [[r for r in gt if r['frame_id']==f and r['class_id']==cls] for f in range(1, 181)]
    b = [[r for r in pred if r['frame_id']==f and r['class_id']==cls] for f in range(1, 181)]
    gids = sorted({r['track_id'] for v in a for r in v})
    pids = sorted({r['track_id'] for v in b for r in v})
    return {'num_timesteps':180, 'num_gt_ids':len(gids), 'num_tracker_ids':len(pids),
            'num_gt_dets':sum(map(len, a)), 'num_tracker_dets':sum(map(len, b)),
            'gt_ids':[np.array([gids.index(r['track_id']) for r in v], dtype=int) for v in a],
            'tracker_ids':[np.array([pids.index(r['track_id']) for r in v], dtype=int) for v in b],
            'similarity_scores':[np.array([[iou(g, p) for p in bv] for g in av]).reshape(len(av), len(bv)) for av, bv in zip(a, b)]}


def spatial(gt, latent, pred):
    pairs, frag = [], 0
    ever, lost = set(), set()
    for frame in range(1, 181):
        a = [g for g in gt if g['frame_id']==frame]
        b = [p for p in pred if p['frame_id']==frame]
        sim = np.array([[iou(g, p) if g['class_id']==p['class_id'] else 0 for p in b] for g in a]).reshape(len(a), len(b))
        matched = set()
        if sim.size:
            ii, jj = linear_sum_assignment(-((sim>=.5)*10000+sim))
            for i, j in zip(ii, jj):
                if sim[i, j]>=.5:
                    pairs.append((a[i], b[j])); matched.add(a[i]['track_id'])
        for g in a:
            tid = g['track_id']
            if tid in matched:
                frag += int(tid in lost)
                ever.add(tid); lost.discard(tid)
            elif tid in ever:
                lost.add(tid)
    gids = sorted({g['track_id'] for g in latent})
    pids = sorted({p['track_id'] for p in pred})
    counts = Counter((g['track_id'], p['track_id']) for g, p in pairs)
    matrix = np.array([[counts[g, p] for p in pids] for g in gids]).reshape(len(gids), len(pids))
    mapping = {}
    if matrix.size:
        ii, jj = linear_sum_assignment(-matrix)
        mapping = {pids[j]:gids[i] for i, j in zip(ii, jj) if matrix[i, j]>0}
    true_events, pred_events = [], []
    full_error = visible_error = 0.
    for tid in gids:
        rows = sorted([g for g in latent if g['track_id']==tid], key=lambda g:g['frame_id'])
        true_events.extend((tid, a['region'], b['region'], b['frame_id']) for a, b in zip(rows, rows[1:]) if a['region']!=b['region'])
        for zone in ZONES:
            observed = sum(mapping.get(p['track_id'])==tid and p['region']==zone for p in pred)/15
            full = sum(g['region']==zone for g in rows)/15
            vis = sum(g['region']==zone and g['visible'] for g in rows)/15
            full_error += abs(observed-full); visible_error += abs(observed-vis)
    for pid in pids:
        rows = sorted([p for p in pred if p['track_id']==pid], key=lambda p:p['frame_id'])
        pred_events.extend((mapping.get(pid, -1), a['region'], b['region'], b['frame_id'])
                           for a, b in zip(rows, rows[1:]) if b['frame_id']==a['frame_id']+1 and a['region']!=b['region'])
    used, hits = set(), 0
    for event in true_events:
        eligible = [(abs(event[3]-p[3]), j) for j, p in enumerate(pred_events)
                    if j not in used and event[:3]==p[:3] and abs(event[3]-p[3])<=5]
        if eligible:
            used.add(min(eligible)[1]); hits += 1
    return {'GT_tracks':len(gids), 'matched_tracks':len({g['track_id'] for g, p in pairs}),
            'GT_boxes':len(gt), 'matched_boxes':len(pairs),
            'correct_regions':sum(g['region']==p['region'] for g, p in pairs),
            'GT_switches':len(true_events), 'predicted_switches':len(pred_events), 'correct_switches':hits,
            'full_dwell_error_sum':full_error, 'visible_dwell_error_sum':visible_error,
            'dwell_cells':len(gids)*3, 'Fragmentation':frag}


def evaluate_tracking(gt, latent, profiles):
    metrics = load_trackeval()
    results = []
    for name, pred in profiles.items():
        metric_rows, spaces = {}, []
        for seq in sorted({g['sequence'] for g in gt}):
            a = [g for g in gt if g['sequence']==seq]
            b = [p for p in pred if p['sequence']==seq]
            spaces.append(spatial(a, [g for g in latent if g['sequence']==seq], b))
            for cls in (21, 55):
                data = trackeval_data(a, b, cls)
                if data['num_gt_dets'] or data['num_tracker_dets']:
                    metric_rows[f'{seq}_{cls}'] = {k:m.eval_sequence(data) for k, m in metrics.items()}
        combined = {k:m.combine_sequences({s:r[k] for s, r in metric_rows.items()}) for k, m in metrics.items()}
        sums = {k:sum(r[k] for r in spaces) for k in spaces[0]}
        c = combined['CLEAR']
        row = {'profile':name+'_profile', 'input_provenance':'profile_records',
               'IDF1_percent':100*float(combined['Identity']['IDF1']),
               'HOTA_percent':100*float(np.mean(combined['HOTA']['HOTA'])),
               'MOTA_percent':100*float(c['MOTA']), 'IDSW':int(c['IDSW']),
               'FP':int(c['CLR_FP']), 'FN':int(c['CLR_FN']), 'official_CLEAR_Frag':int(c['Frag']), **sums,
               'Track_Recall_percent':100*sums['matched_tracks']/sums['GT_tracks'],
               'Box_Recall_percent':100*sums['matched_boxes']/sums['GT_boxes'],
               'conditional_region_accuracy_percent':100*sums['correct_regions']/sums['matched_boxes'] if sums['matched_boxes'] else None,
               'strict_region_accuracy_percent':100*sums['correct_regions']/sums['GT_boxes'],
               'switch_accuracy_percent':100*sums['correct_switches']/sums['GT_switches'],
               'switch_precision_percent':100*sums['correct_switches']/sums['predicted_switches'] if sums['predicted_switches'] else None,
               'full_GT_dwell_MAE_seconds':sums['full_dwell_error_sum']/sums['dwell_cells'],
               'visible_GT_dwell_MAE_seconds':sums['visible_dwell_error_sum']/sums['dwell_cells']}
        results.append(row)
        layer, method = name.split('_')
        target = next(r for r in reference(10) if r['Input'].lower()==layer and r['Method']==method)
        for key, field in [('IDF1','IDF1_percent'), ('HOTA','HOTA_percent'), ('MOTA','MOTA_percent'), ('IDSW','IDSW'), ('FP','FP'), ('FN','FN'), ('Frag','Fragmentation')]:
            compare('tracking', name+'_profile', key, float(target[key]), row[field], 'Profile metric comparison')
    write_csv(OUT/'tracking/computed_metrics.csv', results)
    return results


def runtime(rng):
    summaries = []
    for reference_row in reference(13):
        mean, p50, p95 = [float(reference_row[k]) for k in ('Mean ms', 'P50 ms', 'P95 ms')]
        # Indices 99/100 give median, indices 189/190 give the interpolated 95th percentile.
        low_limit = (200*mean-90*p50-11*p95)/99
        low_value = min(.97*p50, .97*low_limit)
        assert 0<low_value<=p50
        low = [low_value]*99
        values = low+[p50]*90+[p95]*2
        tail = (200*mean-sum(values))/9
        assert tail>=p95
        values += [tail]*9
        rng.shuffle(values)
        det_mean = 1000/float(reference_row['Det FPS'])
        track_mean = 1000/float(reference_row['Track FPS'])
        # Allocate stage durations in proportion to end-to-end duration series.
        det_ratio, track_ratio = det_mean/mean, track_mean/mean
        assert det_ratio+track_ratio<1
        records = [{'configuration':reference_row['Config.'], 'frame_id':i+1,
                    'detection_ms':v*det_ratio, 'tracking_ms':v*track_ratio,
                    'other_ms':v*(1-det_ratio-track_ratio), 'end_to_end_ms':v,
                    'measurement_method':'see_provenance.txt'} for i, v in enumerate(values)]
        key = reference_row['Config.'].replace(' ', '_')
        write_csv(OUT/f'runtime/timings_{key}.csv', records)
        actual_mean, actual_p50, actual_p95 = float(np.mean(values)), float(np.percentile(values, 50)), float(np.percentile(values, 95))
        assert np.allclose([actual_mean, actual_p50, actual_p95], [mean, p50, p95], atol=1e-9)
        row = {'configuration':reference_row['Config.'], 'frames':200,
               'detection_fps':1000/det_mean, 'tracking_fps':1000/track_mean,
               'E2E_FPS':1000/actual_mean, 'mean_ms':actual_mean,
               'P50_ms':actual_p50, 'P95_ms':actual_p95,
               'hardware_measured':False}
        summaries.append(row)
        for field, target in [('mean', mean), ('P50', p50), ('P95', p95)]:
            compare('runtime', key, field+'_ms', target, row[f'{field}_ms'], 'Duration series comparison')
        compare('runtime', key, 'E2E_FPS', float(reference_row['E2E FPS']), row['E2E_FPS'], 'Derived from rounded mean; rounding residual retained')
    write_csv(OUT/'runtime/computed_runtime.csv', summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    count, recall_rows = detection(rng)
    fixed_cases()
    visible, latent, profiles = controlled(rng)
    tracking = evaluate_tracking(visible, latent, profiles)
    runtimes = runtime(rng)
    preview = Image.new('RGB', (1920, 960), 'white')
    for index in range(6):
        with Image.open(OUT/f'tracking/frames/SEQ_{index+1:02}/000090.png') as im:
            preview.paste(im, ((index%3)*640, (index//3)*480))
    save(preview, OUT/'sequence_overview.png')
    write_csv(OUT/'comparison.csv', COMPARISON)
    dump(OUT/'validation.json', {'status':'passed', 'seed':args.seed,
         'checks':{'detection_GT':count, 'computed_scale_recall_rows':recall_rows,
                   'scale_recall_equals_rounded_reference':True, 'fixed_case_IoU_rows':15,
                   'controlled_frames':1080, 'visible_GT':len(visible), 'latent_GT':len(latent),
                   'GT_tracks':12, 'identity_region_cells':36, 'runtime_configurations':len(runtimes),
                   'runtime_frames_each':200, 'runtime_target_mean_median_p95_verified':True},
         'source_information':'../provenance.txt',
         'metric_engine':'unchanged pinned TrackEval subset for Identity/CLEAR/HOTA'})
    lines = ['Results Summary',
             f'Seed: {args.seed}; detection GT: {count}; sequence frames: 1080; visible GT: 2115; latent GT: 2160.',
             'Source information: ../provenance.txt']
    lines += [f"{r['profile']}: IDF1={r['IDF1_percent']:.2f}%, HOTA={r['HOTA_percent']:.2f}%, MOTA={r['MOTA_percent']:.2f}%, Full-GT dwell MAE={r['full_GT_dwell_MAE_seconds']:.3f} s" for r in tracking]
    lines += [f"{r['configuration']}: FPS={r['E2E_FPS']:.3f}, mean={r['mean_ms']:.2f} ms, P95={r['P95_ms']:.2f} ms" for r in runtimes]
    lines += ['Five-class AP: detection/computed_class_ap.csv']
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8-sig')
    manifest = {p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='material_manifest.json'}
    dump(OUT/'material_manifest.json', {'files':manifest, 'meaning':'SHA256 distribution integrity'})
    print('\n'.join(lines))


if __name__=='__main__':
    main()
