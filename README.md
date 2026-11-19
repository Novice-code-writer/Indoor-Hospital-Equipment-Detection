# Indoor-Hospital-Equipment-Detection
Code and materials for hospital equipment detection metrics, trajectory evaluation, 2D region mapping, and latency analysis, with reference tables and documented data provenance.

# File and Module Reference

All paths are relative to this README's directory. Filename patterns group files that serve the same function.

See `provenance.txt` for material sources and usage conditions.

## Entry Point and Dependencies

| File | Purpose |
| --- | --- |
| `run_pipeline.py` | Main entry point for image and record generation, metric calculation, trajectory evaluation, region statistics, runtime summaries, and checksum generation. |
| `requirements.txt` | Python dependency requirements for NumPy, SciPy, and Pillow. |
| [Documentation text file](provenance.txt) | Material scope, usage instructions, calculation conventions, and provenance. |

## Detection Module

| File or Directory | Purpose |
| --- | --- |
| `generated/detection/images/` | Detection images linked to annotation and prediction records through `image_id`. |
| `generated/detection/ground_truth.csv` | Image IDs, object IDs, classes, size groups, bounding boxes, and image dimensions. |
| `generated/detection/predictions_B0-full.csv` | Candidate boxes and confidence scores for the B0-full configuration. |
| `generated/detection/predictions_A2.csv` | Candidate boxes and confidence scores for the A2 configuration. |
| `generated/detection/predictions_A4.csv` | Candidate boxes and confidence scores for the A4 configuration. |
| `generated/detection/computed_scale_recall.csv` | Ground-truth counts, true-positive counts, and recall by configuration, class, and size group, with confidence and matching thresholds. |
| `generated/detection/computed_class_ap.csv` | AP50, AP50-95, and five-class mean values for the detection records. |

## Fixed Cases Module

| File | Purpose |
| --- | --- |
| `generated/fixed_cases/CASE_Q01.png` | Hospital bed case with bounding-box overlays. |
| `generated/fixed_cases/CASE_Q02.png` | Bedside monitor case with bounding-box overlays. |
| `generated/fixed_cases/CASE_Q03.png` | Infusion pump case with bounding-box overlays. |
| `generated/fixed_cases/CASE_Q04.png` | Ventilator case with bounding-box overlays. |
| `generated/fixed_cases/CASE_Q05.png` | Wheelchair case with bounding-box overlays. |
| `generated/fixed_cases/computed_iou.csv` | Bounding-box coordinates, relative object areas, and computed IoUs for five cases across three configurations. |

## Trajectory and Region Modules

| File or Directory | Purpose |
| --- | --- |
| `generated/tracking/frames/SEQ_01/` | Translation sequence frames. |
| `generated/tracking/frames/SEQ_02/` | Scale variation sequence frames. |
| `generated/tracking/frames/SEQ_03/` | Partial occlusion sequence frames. |
| `generated/tracking/frames/SEQ_04/` | Short disappearance sequence frames. |
| `generated/tracking/frames/SEQ_05/` | Re-entry sequence frames. |
| `generated/tracking/frames/SEQ_06/` | Same-class crossing sequence frames. |
| `generated/tracking/latent_gt.csv` | Full trajectory references, retaining position, identity, region, and visibility during invisible intervals. |
| `generated/tracking/visible_gt.csv` | Visible-object annotations for box matching and trajectory evaluation. |
| `generated/tracking/oracle_FrameIoU_profile.csv` | Trajectory records for the FrameIoU behavior profile under Oracle input conditions. |
| `generated/tracking/oracle_ByteTrack_profile.csv` | Trajectory records for the ByteTrack behavior profile under Oracle input conditions. |
| `generated/tracking/e2e_FrameIoU_profile.csv` | Trajectory records for the FrameIoU behavior profile under E2E input conditions. |
| `generated/tracking/e2e_ByteTrack_profile.csv` | Trajectory records for the ByteTrack behavior profile under E2E input conditions. |
| `generated/tracking/computed_metrics.csv` | IDF1, HOTA, MOTA, identity switches, false positives, false negatives, fragmentation, coverage, region accuracy, transition metrics, and dwell-time errors. |

## Runtime Statistics Module

| File | Purpose |
| --- | --- |
| `generated/runtime/timings_CPU_640_FP32.csv` | Per-frame duration series for the CPU, 640-input, FP32 configuration. |
| `generated/runtime/timings_GPU_640_FP32.csv` | Per-frame duration series for the GPU, 640-input, FP32 configuration. |
| `generated/runtime/timings_GPU_640_FP16.csv` | Per-frame duration series for the GPU, 640-input, FP16 configuration. |
| `generated/runtime/timings_GPU_512_FP16.csv` | Per-frame duration series for the GPU, 512-input, FP16 configuration. |
| `generated/runtime/timings_GPU_416_FP16.csv` | Per-frame duration series for the GPU, 416-input, FP16 configuration. |
| `generated/runtime/computed_runtime.csv` | Stage FPS, end-to-end FPS, mean latency, P50 latency, and P95 latency for all five configurations. |

## Manuscript Table References

| File | Corresponding Content |
| --- | --- |
| `reference/reported_table_01.csv` | Table 1: Data sources and task input sizes. |
| `reference/reported_table_02.csv` | Table 2: Valid annotation counts for five equipment classes across dataset partitions. |
| `reference/reported_table_03.csv` | Table 3: Three model configurations and shared training settings. |
| `reference/reported_table_04.csv` | Table 4: Controlled trajectory conditions and ground-truth counts. |
| `reference/reported_table_05.csv` | Table 5: Nursing demonstration video windows and processed frame counts. |
| `reference/reported_table_06.csv` | Table 6: Overall validation and test detection metrics for three models. |
| `reference/reported_table_07.csv` | Table 7: Equipment class AP and recall at a fixed operating point. |
| `reference/reported_table_08.csv` | Table 8: Ground-truth counts and recall by class and size group. |
| `reference/reported_table_09.csv` | Table 9: Relative areas and IoUs of fixed cases. |
| `reference/reported_table_10.csv` | Table 10: Tracking metrics for two input conditions and two association configurations. |
| `reference/reported_table_11.csv` | Table 11: Trajectory coverage, region accuracy, transition metrics, and dwell-time errors. |
| `reference/reported_table_12.csv` | Table 12: Bed outputs and wheelchair candidate statistics in nursing demonstration videos. |
| `reference/reported_table_13.csv` | Table 13: Throughput, latency, and validation accuracy for five deployment configurations. |

## Summaries and Validation

| File | Purpose |
| --- | --- |
| `generated/comparison.csv` | Reference values, computed values, differences, and calculation conditions. |
| `generated/summary.txt` | Run sizes and key metric summaries. |
| `generated/validation.json` | Checks for record counts, matching results, geometry constraints, and latency statistics. |
| `generated/material_manifest.json` | Relative output paths and SHA256 checksums. |
| `generated/sequence_overview.png` | Overview of the six trajectory conditions. |

## Evaluation Dependencies

| File | Purpose |
| --- | --- |
| `vendor/trackeval/metrics/identity.py` | Identity metrics, including IDF1. |
| `vendor/trackeval/metrics/clear.py` | CLEAR metrics, including MOTA, IDSW, FP, FN, and Frag. |
| `vendor/trackeval/metrics/hota.py` | HOTA metric evaluation. |
| `vendor/trackeval/metrics/_base_metric.py` | Shared base class for evaluation metrics. |
| `vendor/trackeval/utils.py` | Evaluation configuration and utility functions. |
| `vendor/trackeval/_timing.py` | Timing decorators for evaluation functions. |
| `vendor/TrackEval_LICENSE` | TrackEval license. |
| `provenance.txt` | Material sources, usage conditions, dependency revision, and compatibility notes. |

## Main Program Functions

| Function | Module Responsibility |
| --- | --- |
| `detection()` | Detection images, box records, size-specific recall, and AP. |
| `fixed_cases()` | Fixed cases and IoU calculations. |
| `controlled()` | Six sequence conditions, annotations, and trajectory profile records. |
| `evaluate_tracking()` | Trajectory evaluation and region metric aggregation. |
| `spatial()` | Box matching, global identity mapping, region transitions, and dwell-time errors. |
| `trackeval_data()` | TrackEval input preparation by sequence and class. |
| `load_trackeval()` | Loading of Identity, CLEAR, and HOTA modules. |
| `runtime()` | Per-frame duration series and throughput and latency summaries. |
| `matches()` | Class-aware, confidence-ordered, one-to-one box matching. |
| `ap101()` | AP calculation using 101 recall points. |
| `iou()` | Bounding-box intersection over union. |
| `region()` | Mapping of bounding-box bottom-center points to region labels. |
| `compare()` | Reference and computed value comparison records. |
| `main()` | Module orchestration, summaries, validation records, and output checksums. |
