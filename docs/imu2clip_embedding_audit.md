# IMU2CLIP Embedding Audit

- manifest: `/data/cache/manifests/pretrain_full_cached_urls.txt`
- checkpoint: `/artifacts/checkpoints/imu2clip_style/validation_imu2clip_style_encoder.pt`
- clips embedded: `500`
- effective_rank: `31.392`
- mean_pairwise_cosine: `0.969`
- rank_correlation_vs_window_mean_std_pool: `0.232`
- passed_effective_rank_gate: `True`

## Window Sensitivity

```json
{
  "cosine_2s_vs_10s_mean": 0.9710189759731293,
  "cosine_2s_vs_5s_mean": 0.9819938266277313
}
```

## Nearest Neighbors

```json
[
  {
    "neighbors": [
      {
        "clip_id": "clip002",
        "distance": 0.011229872843502742
      },
      {
        "clip_id": "clip014",
        "distance": 0.011263092548633002
      },
      {
        "clip_id": "clip004",
        "distance": 0.011291754517547492
      },
      {
        "clip_id": "clip010",
        "distance": 0.011333367576896736
      },
      {
        "clip_id": "clip020",
        "distance": 0.011667256209658694
      }
    ],
    "query_clip_id": "clip001"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip011",
        "distance": 0.010093959973479394
      },
      {
        "clip_id": "clip001",
        "distance": 0.010293300726191279
      },
      {
        "clip_id": "clip004",
        "distance": 0.010528659192653445
      },
      {
        "clip_id": "clip016",
        "distance": 0.010580627698647649
      },
      {
        "clip_id": "clip010",
        "distance": 0.010771012429176663
      }
    ],
    "query_clip_id": "clip002"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip015",
        "distance": 0.009598129530198896
      },
      {
        "clip_id": "clip016",
        "distance": 0.009671064531202811
      },
      {
        "clip_id": "clip001",
        "distance": 0.009715230577962086
      },
      {
        "clip_id": "clip002",
        "distance": 0.009863460383109812
      },
      {
        "clip_id": "clip014",
        "distance": 0.010026264487668302
      }
    ],
    "query_clip_id": "clip003"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip001",
        "distance": 0.00976972102607021
      },
      {
        "clip_id": "clip002",
        "distance": 0.010135619499930026
      },
      {
        "clip_id": "clip014",
        "distance": 0.010250836096905691
      },
      {
        "clip_id": "clip015",
        "distance": 0.010394628863890754
      },
      {
        "clip_id": "clip005",
        "distance": 0.01045034290454816
      }
    ],
    "query_clip_id": "clip004"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip013",
        "distance": 0.010296455979097452
      },
      {
        "clip_id": "clip019",
        "distance": 0.010343949425698495
      },
      {
        "clip_id": "clip012",
        "distance": 0.010420763408828693
      },
      {
        "clip_id": "clip004",
        "distance": 0.010421190605752972
      },
      {
        "clip_id": "clip004",
        "distance": 0.01045034290454816
      }
    ],
    "query_clip_id": "clip005"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip007",
        "distance": 0.010228642214476169
      },
      {
        "clip_id": "clip007",
        "distance": 0.01074383218885755
      },
      {
        "clip_id": "clip020",
        "distance": 0.010793238007732464
      },
      {
        "clip_id": "clip016",
        "distance": 0.011136462560008442
      },
      {
        "clip_id": "clip019",
        "distance": 0.011346784529223886
      }
    ],
    "query_clip_id": "clip006"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip006",
        "distance": 0.010228642214476169
      },
      {
        "clip_id": "clip008",
        "distance": 0.010692331430347757
      },
      {
        "clip_id": "clip014",
        "distance": 0.011077082821504103
      },
      {
        "clip_id": "clip020",
        "distance": 0.011150928250366454
      },
      {
        "clip_id": "clip010",
        "distance": 0.011203466600081757
      }
    ],
    "query_clip_id": "clip007"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip007",
        "distance": 0.010692331430347757
      },
      {
        "clip_id": "clip005",
        "distance": 0.010767188172816966
      },
      {
        "clip_id": "clip010",
        "distance": 0.01086406221634495
      },
      {
        "clip_id": "clip009",
        "distance": 0.011106334061941392
      },
      {
        "clip_id": "clip015",
        "distance": 0.011207649598559022
      }
    ],
    "query_clip_id": "clip008"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip008",
        "distance": 0.011106334061941392
      },
      {
        "clip_id": "clip001",
        "distance": 0.011120968317263058
      },
      {
        "clip_id": "clip002",
        "distance": 0.011174724657086355
      },
      {
        "clip_id": "clip011",
        "distance": 0.011485808484690274
      },
      {
        "clip_id": "clip019",
        "distance": 0.011694027006727281
      }
    ],
    "query_clip_id": "clip009"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip001",
        "distance": 0.010945786384869094
      },
      {
        "clip_id": "clip002",
        "distance": 0.011159708068891216
      },
      {
        "clip_id": "clip002",
        "distance": 0.011280747102911048
      },
      {
        "clip_id": "clip001",
        "distance": 0.011333367576896736
      },
      {
        "clip_id": "clip004",
        "distance": 0.011362016354246052
      }
    ],
    "query_clip_id": "clip010"
  }
]
```
