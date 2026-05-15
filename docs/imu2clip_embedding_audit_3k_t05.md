# IMU2CLIP Embedding Audit

- manifest: `/data/cache/manifests/pretrain_full_cached_urls.txt`
- checkpoint: `/artifacts/checkpoints/imu2clip_style/validation_imu2clip_style_encoder_3k_t05.pt`
- clips embedded: `500`
- effective_rank: `28.112`
- mean_pairwise_cosine: `0.967`
- rank_correlation_vs_window_mean_std_pool: `0.617`
- passed_effective_rank_gate: `True`

## Window Sensitivity

```json
{
  "cosine_2s_vs_10s_mean": 0.9686516869068146,
  "cosine_2s_vs_5s_mean": 0.983253116607666
}
```

## Nearest Neighbors

```json
[
  {
    "neighbors": [
      {
        "clip_id": "clip020",
        "distance": 0.011210235063582541
      },
      {
        "clip_id": "clip012",
        "distance": 0.011275780011857361
      },
      {
        "clip_id": "clip020",
        "distance": 0.011405443529599824
      },
      {
        "clip_id": "clip014",
        "distance": 0.011537401880874043
      },
      {
        "clip_id": "clip009",
        "distance": 0.01162674868235003
      }
    ],
    "query_clip_id": "clip001"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip006",
        "distance": 0.010346380960993384
      },
      {
        "clip_id": "clip013",
        "distance": 0.011043471572358854
      },
      {
        "clip_id": "clip008",
        "distance": 0.011231068182628645
      },
      {
        "clip_id": "clip007",
        "distance": 0.011288746113302883
      },
      {
        "clip_id": "clip005",
        "distance": 0.012016517703809737
      }
    ],
    "query_clip_id": "clip002"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip002",
        "distance": 0.009296982224225703
      },
      {
        "clip_id": "clip005",
        "distance": 0.009447480245375894
      },
      {
        "clip_id": "clip006",
        "distance": 0.009460414751190083
      },
      {
        "clip_id": "clip015",
        "distance": 0.009534286399545677
      },
      {
        "clip_id": "clip016",
        "distance": 0.009576850050490693
      }
    ],
    "query_clip_id": "clip003"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip001",
        "distance": 0.010015770469822494
      },
      {
        "clip_id": "clip005",
        "distance": 0.010425602817575319
      },
      {
        "clip_id": "clip001",
        "distance": 0.010760389868990083
      },
      {
        "clip_id": "clip020",
        "distance": 0.010862945628212994
      },
      {
        "clip_id": "clip015",
        "distance": 0.010966948819514544
      }
    ],
    "query_clip_id": "clip004"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip008",
        "distance": 0.008932673845318817
      },
      {
        "clip_id": "clip015",
        "distance": 0.008976717203066142
      },
      {
        "clip_id": "clip011",
        "distance": 0.009230148071568411
      },
      {
        "clip_id": "clip017",
        "distance": 0.009670412089762825
      },
      {
        "clip_id": "clip019",
        "distance": 0.009672992188610507
      }
    ],
    "query_clip_id": "clip005"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip003",
        "distance": 0.009733320149692637
      },
      {
        "clip_id": "clip002",
        "distance": 0.010346380960993384
      },
      {
        "clip_id": "clip010",
        "distance": 0.010378747989909454
      },
      {
        "clip_id": "clip006",
        "distance": 0.010527093984888714
      },
      {
        "clip_id": "clip020",
        "distance": 0.010628671723639016
      }
    ],
    "query_clip_id": "clip006"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip007",
        "distance": 0.010785210798414102
      },
      {
        "clip_id": "clip005",
        "distance": 0.010848791118335921
      },
      {
        "clip_id": "clip020",
        "distance": 0.010950123484943841
      },
      {
        "clip_id": "clip007",
        "distance": 0.011153894358601146
      },
      {
        "clip_id": "clip008",
        "distance": 0.011161886275514843
      }
    ],
    "query_clip_id": "clip007"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip005",
        "distance": 0.008932673845318817
      },
      {
        "clip_id": "clip015",
        "distance": 0.009029401758937072
      },
      {
        "clip_id": "clip018",
        "distance": 0.009531852355831871
      },
      {
        "clip_id": "clip012",
        "distance": 0.009605847267124767
      },
      {
        "clip_id": "clip001",
        "distance": 0.009629193286483595
      }
    ],
    "query_clip_id": "clip008"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip008",
        "distance": 0.010515642459288377
      },
      {
        "clip_id": "clip015",
        "distance": 0.010716960594758795
      },
      {
        "clip_id": "clip005",
        "distance": 0.0109425829740194
      },
      {
        "clip_id": "clip005",
        "distance": 0.011032098571228888
      },
      {
        "clip_id": "clip003",
        "distance": 0.011263661975215444
      }
    ],
    "query_clip_id": "clip009"
  },
  {
    "neighbors": [
      {
        "clip_id": "clip012",
        "distance": 0.011492960862851564
      },
      {
        "clip_id": "clip008",
        "distance": 0.012121210951069328
      },
      {
        "clip_id": "clip014",
        "distance": 0.012190217284971827
      },
      {
        "clip_id": "clip005",
        "distance": 0.012248579232686518
      },
      {
        "clip_id": "clip013",
        "distance": 0.01237694559623459
      }
    ],
    "query_clip_id": "clip010"
  }
]
```
