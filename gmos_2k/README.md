# GMOS-2K

A video moving-object-segmentation dataset with **per-object temporal motion labels**, built on top of five existing VOS datasets. Beyond ground-truth masks for moving objects, GMOS-2K specifies the exact temporal windows during which each object is in motion, enabling precise training and evaluation of time-sensitive motion segmentation methods.

## Download

#### Video sources

GMOS-2K reuses RGB videos from five existing datasets. Download them from their original sources:

| Dataset | Link |
|---|---|
| DAVIS17    | [davischallenge.org/davis2017/code.html](https://davischallenge.org/davis2017/code.html) |
| YTVOS19    | [youtube-vos.org/dataset/vos/](https://youtube-vos.org/dataset/vos/) |
| OVIS       | [songbai.site/ovis/index.html#download](https://songbai.site/ovis/index.html#download) |
| MoCA-Mask  | [opendatalab.com/OpenDataLab/MoCA-Mask](https://opendatalab.com/OpenDataLab/MoCA-Mask/cli/main) |
| HOI4D      | [hoi4d.github.io](https://hoi4d.github.io/) |

#### Annotations and temporally fine-grained motion annotations

GMOS-2K annotations (per-frame masks of moving objects + per-object motion frame intervals): [Link](https://drive.google.com/file/d/1HIBtuASf_iLuTPQ1C3VdqqvTWczVTN7w/view?usp=sharing).

> **Note.** Sequence names under `annotations/<subset>/<split>/<seq>/` correspond to the original sequence names in each source dataset above, so masks can be paired directly with the original RGB videos.


## Layout

```
gmos_2k/
├── annotations/
│   ├── davis/      {train, test}/<seq>/{00000.png, …}
│   ├── ytvos/      {train, test}/<seq>/{00000.png, …}
│   ├── ovis/       train/<seq>/{00000.png, …}
│   ├── moca_mask/  train/<seq>/{00000.png, …}
│   └── hoi4d/      train/<seq>/{00000.png, …}
└── time_annotation.csv
```

- `annotations/<subset>/<split>/<seq>/<frame>.png` — palette-indexed PNG masks. Object IDs are remapped to be contiguous starting from 1 across moving objects only; static objects are removed.
- `time_annotation.csv` — one row per sequence with the following columns:

| column | meaning |
|---|---|
| `seq_name` | sequence identifier |
| `time_anno` | `{object_id: [[start_frame, end_frame], …]}` literal dict, listing the frame intervals during which each object is in motion |
| `ann_frames` | number of annotated frames |
| `vid_frames` | total video length in frames |
| `dataset` | one of `davis`, `ytvos`, `ovis`, `moca_mask`, `hoi4d` |
| `split` | `train` or `test` |


## Statistics

| Split | Subset | Videos | Annotated frames | Objects | Objects / video | Motion proportion |
|---|---|---:|---:|---:|---:|---:|
| Train | DAVIS17     |    44 |   3,101 |    92 | 2.09 | 94.8% |
| Train | YTVOS19     | 1,183 |  31,327 | 1,623 | 1.37 | 91.1% |
| Train | OVIS        |   404 |  25,925 | 1,865 | 4.62 | 92.6% |
| Train | MoCA-Mask   |    26 |   1,978 |    26 | 1.00 | 54.8% |
| Train | HOI4D       |   273 |  81,900 |   609 | 2.23 | 69.9% |
| **Total (Train)** | | **1,930** | **144,231** | **4,215** | **2.18** | **86.4%** |
| Test  | DAVIS17     |    19 |   1,237 |    31 | 1.63 | 96.9% |
| Test  | YTVOS19     |   261 |   7,115 |   402 | 1.54 | 91.5% |
| **Total (Test)**  | | **280** | **8,352** | **433** | **1.55** | **91.9%** |

Composition spans complementary domains: heavily occluded scenes (OVIS), hand-object interactions (HOI4D), wildlife and human-centric clips (DAVIS17, YTVOS19, MoCA-Mask).


## DAVIS17-IM and YTVOS19-IM (MOS-I test splits)

For the MOS-I evaluation protocol, two test splits are derived directly from GMOS-2K:

- **DAVIS17-IM**: the 19 DAVIS17 test sequences.
- **YTVOS19-IM**: the 261 YTVOS19 test sequences with non-empty motion intervals (231 after removing fully-static sequences).

These splits are evaluated by `metrics/benchmark_mos_i.py` (see the [main README](../README.md#evaluation)).


## Citation

If you use GMOS-2K, please cite the paper:

```bibtex
@article{xie2026gmos,
    title     = {GMOS: Grounding Moving Object Segmentation in 3D Space and Time},
    author    = {Junyu Xie and Tengda Han and Weidi Xie and Andrew Zisserman},
    journal   = {arXiv preprint arXiv:xxxx.xxxxx},
    year      = {2026}
}
```

The five constituent subsets retain their original licenses; please cite the source datasets as well: DAVIS17, YTVOS19, OVIS, MoCA-Mask, and HOI4D.
