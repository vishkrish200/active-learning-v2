# ImageBind IMU Embedding Audit

- run_id: `imagebind-imu-621a6f03d709`
- manifest: `/data/cache/manifests/pretrain_full_cached_urls.txt`
- clips embedded: `500`
- effective_rank: `20.932`
- mean_pairwise_cosine: `0.818`
- rank_correlation_vs_window_mean_std_pool: `0.627`
- passed_effective_rank_gate: `True`
- passed_cosine_gate: `False`
- passed_embedding_audit: `False`

## Window Contract

ImageBind IMU is fixed at `(B, 6, 2000)`, i.e. 10 seconds at 200 Hz. Long clips are resampled, normalized per clip/per channel, windowed with 50% overlap, max pooled, and L2 normalized.

```json
{
  "expected_hz": 200,
  "input_shape": "(B, 6, 2000)",
  "pool": "max",
  "window_len_s": 10.0
}
```

## Nearest Neighbors

```json
[
  {
    "neighbors": [
      {
        "clip_id": "clip004",
        "distance": 0.05111934580712174
      },
      {
        "clip_id": "clip005",
        "distance": 0.058093021893754426
      },
      {
        "clip_id": "clip015",
        "distance": 0.05906962525157122
      },
      {
        "clip_id": "clip011",
        "distance": 0.06127468473265929
      },
      {
        "clip_id": "clip008",
        "distance": 0.0626475187456138
      }
    ],
    "query_clip_id": "clip001"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip012",
        "distance": 0.04488102745892686
      },
      {
        "clip_id": "clip003",
        "distance": 0.049273470499406
      },
      {
        "clip_id": "clip016",
        "distance": 0.053937714244832846
      },
      {
        "clip_id": "clip011",
        "distance": 0.05513994768273378
      },
      {
        "clip_id": "clip007",
        "distance": 0.060369699906323016
      }
    ],
    "query_clip_id": "clip002"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip019",
        "distance": 0.030366265031158024
      },
      {
        "clip_id": "clip016",
        "distance": 0.03876669485382922
      },
      {
        "clip_id": "clip010",
        "distance": 0.042141246382247144
      },
      {
        "clip_id": "clip007",
        "distance": 0.044507944857391624
      },
      {
        "clip_id": "clip006",
        "distance": 0.047949137572926626
      }
    ],
    "query_clip_id": "clip003"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip001",
        "distance": 0.031466142961598664
      },
      {
        "clip_id": "clip017",
        "distance": 0.03275259119565743
      },
      {
        "clip_id": "clip014",
        "distance": 0.03414356208885172
      },
      {
        "clip_id": "clip006",
        "distance": 0.03559374775516111
      },
      {
        "clip_id": "clip016",
        "distance": 0.03747747576228866
      }
    ],
    "query_clip_id": "clip004"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip013",
        "distance": 0.051006720684955376
      },
      {
        "clip_id": "clip007",
        "distance": 0.05191138479094859
      },
      {
        "clip_id": "clip012",
        "distance": 0.05206021128838034
      },
      {
        "clip_id": "clip004",
        "distance": 0.054637348212246195
      },
      {
        "clip_id": "clip016",
        "distance": 0.05471139763856736
      }
    ],
    "query_clip_id": "clip005"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip017",
        "distance": 0.034982978631113326
      },
      {
        "clip_id": "clip004",
        "distance": 0.03559374775516111
      },
      {
        "clip_id": "clip009",
        "distance": 0.03684802244822083
      },
      {
        "clip_id": "clip014",
        "distance": 0.03707089135926267
      },
      {
        "clip_id": "clip009",
        "distance": 0.04580750563440983
      }
    ],
    "query_clip_id": "clip006"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip001",
        "distance": 0.04130901540192855
      },
      {
        "clip_id": "clip004",
        "distance": 0.041962644208820565
      },
      {
        "clip_id": "clip008",
        "distance": 0.044599062321828864
      },
      {
        "clip_id": "clip016",
        "distance": 0.045905952460849164
      },
      {
        "clip_id": "clip012",
        "distance": 0.04752301247704771
      }
    ],
    "query_clip_id": "clip007"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip001",
        "distance": 0.0268909603666887
      },
      {
        "clip_id": "clip013",
        "distance": 0.031279730439782605
      },
      {
        "clip_id": "clip012",
        "distance": 0.034203935511608496
      },
      {
        "clip_id": "clip015",
        "distance": 0.03514968183953171
      },
      {
        "clip_id": "clip018",
        "distance": 0.036209627617654117
      }
    ],
    "query_clip_id": "clip008"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip006",
        "distance": 0.03684802244822083
      },
      {
        "clip_id": "clip010",
        "distance": 0.040866926379921065
      },
      {
        "clip_id": "clip014",
        "distance": 0.042567017744878366
      },
      {
        "clip_id": "clip007",
        "distance": 0.04426500965666491
      },
      {
        "clip_id": "clip017",
        "distance": 0.04438931763990839
      }
    ],
    "query_clip_id": "clip009"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip008",
        "distance": 0.0345820944719204
      },
      {
        "clip_id": "clip002",
        "distance": 0.03808563079738336
      },
      {
        "clip_id": "clip009",
        "distance": 0.040866926379921065
      },
      {
        "clip_id": "clip002",
        "distance": 0.04336497252617999
      },
      {
        "clip_id": "clip007",
        "distance": 0.048737622114091605
      }
    ],
    "query_clip_id": "clip010"
  }
]
```
