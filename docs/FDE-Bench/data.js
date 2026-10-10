// Source: benchmark/arxiv/5_experiments.tex (Table 1), 8_appendix.tex (case inventory), and main.tex.
// quality is a normalized score; all other result metrics are percentages. Zero values are retained.
// Case family codes: CO = combinatorial optimization; ML = machine learning.
window.FDE_DATA = {
  "results": [
    {
      "model": "GPT-6-Astra",
      "scaffold": "opencode",
      "quality": 0.5459,
      "dsr": 5.44,
      "pass3": 91.84,
      "all3": 81.63,
      "recall": 83.64,
      "precision": 36.53,
      "askF1": 48.82
    },
    {
      "model": "GLM-5.3",
      "scaffold": "opencode",
      "quality": 0.5413,
      "dsr": 1.36,
      "pass3": 91.84,
      "all3": 77.55,
      "recall": 66.83,
      "precision": 39.7,
      "askF1": 47.46
    },
    {
      "model": "Qwen3.8-Max",
      "scaffold": "opencode",
      "quality": 0.5387,
      "dsr": 2.72,
      "pass3": 95.92,
      "all3": 73.47,
      "recall": 70.56,
      "precision": 35.41,
      "askF1": 45.11
    },
    {
      "model": "Grok-4.6",
      "scaffold": "opencode",
      "quality": 0.5221,
      "dsr": 2.72,
      "pass3": 87.76,
      "all3": 79.59,
      "recall": 68.19,
      "precision": 38.3,
      "askF1": 46.42
    },
    {
      "model": "DeepSeek-V4-Pro",
      "scaffold": "opencode",
      "quality": 0.4986,
      "dsr": 1.36,
      "pass3": 91.84,
      "all3": 71.43,
      "recall": 55.08,
      "precision": 37.93,
      "askF1": 42.73
    },
    {
      "model": "Kimi-K3",
      "scaffold": "opencode",
      "quality": 0.4187,
      "dsr": 0.0,
      "pass3": 91.84,
      "all3": 53.06,
      "recall": 55.03,
      "precision": 37.6,
      "askF1": 43.27
    },
    {
      "model": "Gemini-3.1-Pro",
      "scaffold": "opencode",
      "quality": 0.3873,
      "dsr": 0.0,
      "pass3": 89.8,
      "all3": 59.18,
      "recall": 47.1,
      "precision": 39.48,
      "askF1": 40.38
    },
    {
      "model": "GPT-6-Astra",
      "scaffold": "openhands",
      "quality": 0.5302,
      "dsr": 4.76,
      "pass3": 93.88,
      "all3": 67.35,
      "recall": 84.67,
      "precision": 36.8,
      "askF1": 49.5
    },
    {
      "model": "GLM-5.3",
      "scaffold": "openhands",
      "quality": 0.5274,
      "dsr": 2.04,
      "pass3": 91.84,
      "all3": 67.35,
      "recall": 67.12,
      "precision": 36.79,
      "askF1": 45.48
    },
    {
      "model": "Grok-4.6",
      "scaffold": "openhands",
      "quality": 0.5159,
      "dsr": 0.68,
      "pass3": 91.84,
      "all3": 71.43,
      "recall": 69.91,
      "precision": 36.95,
      "askF1": 45.91
    },
    {
      "model": "DeepSeek-V4-Pro",
      "scaffold": "openhands",
      "quality": 0.5104,
      "dsr": 0.68,
      "pass3": 93.88,
      "all3": 71.43,
      "recall": 65.24,
      "precision": 36.41,
      "askF1": 44.92
    },
    {
      "model": "Qwen3.8-Max",
      "scaffold": "openhands",
      "quality": 0.5071,
      "dsr": 6.12,
      "pass3": 93.88,
      "all3": 65.31,
      "recall": 75.37,
      "precision": 30.89,
      "askF1": 42.24
    },
    {
      "model": "Gemini-3.1-Pro",
      "scaffold": "openhands",
      "quality": 0.4571,
      "dsr": 0.0,
      "pass3": 87.76,
      "all3": 65.31,
      "recall": 45.45,
      "precision": 37.77,
      "askF1": 39.25
    },
    {
      "model": "Kimi-K3",
      "scaffold": "openhands",
      "quality": 0.4119,
      "dsr": 0.68,
      "pass3": 85.71,
      "all3": 57.14,
      "recall": 57.94,
      "precision": 35.39,
      "askF1": 41.34
    },
    {
      "model": "GPT-6-Astra",
      "scaffold": "codex",
      "quality": 0.551,
      "dsr": 4.08,
      "pass3": 93.88,
      "all3": 75.51,
      "recall": 84.7,
      "precision": 37.87,
      "askF1": 50.12
    },
    {
      "model": "DeepSeek-V4-Pro",
      "scaffold": "deepseek-harness",
      "quality": 0.5271,
      "dsr": 1.36,
      "pass3": 91.84,
      "all3": 75.51,
      "recall": 61.79,
      "precision": 39.32,
      "askF1": 45.93
    },
    {
      "model": "Kimi-K3",
      "scaffold": "kimi",
      "quality": 0.4203,
      "dsr": 0.68,
      "pass3": 89.8,
      "all3": 59.18,
      "recall": 55.69,
      "precision": 36.69,
      "askF1": 42.9
    },
    {
      "model": "Gemini-3.1-Pro",
      "scaffold": "gemini",
      "quality": 0.4037,
      "dsr": 0.0,
      "pass3": 87.76,
      "all3": 57.14,
      "recall": 44.94,
      "precision": 38.82,
      "askF1": 39.37
    }
  ],
  "cases": [
    {
      "id": "01",
      "name": "Vehicle-drag prediction",
      "family": "ML",
      "objective": "Mean relative drag-coefficient error (Cd_MRE)",
      "constraints": "Coverage, unique finite rows; 0.05 ≤ Cd ≤ 1.5"
    },
    {
      "id": "02",
      "name": "City delivery routing",
      "family": "CO",
      "objective": "combined route cost (distance/time/penalties)",
      "constraints": "all required stops, feasible routes and valid timestamps"
    },
    {
      "id": "03",
      "name": "Crop-yield prediction",
      "family": "ML",
      "objective": "yearly mean RMSE",
      "constraints": "required predictions, finite values and complete test coverage"
    },
    {
      "id": "04",
      "name": "Electricity-price forecasting",
      "family": "ML",
      "objective": "weighted MAE/p95/p99/high-price underprediction error",
      "constraints": "complete horizon, finite values and aligned timestamps"
    },
    {
      "id": "05",
      "name": "Flight-booking forecasting",
      "family": "ML",
      "objective": "seat-booking forecast score",
      "constraints": "complete horizon, finite nonnegative forecasts"
    },
    {
      "id": "06",
      "name": "Ion-beam uniformity",
      "family": "ML",
      "objective": "mean WAPE-derived uniformity score",
      "constraints": "all required rows, finite predictions and valid bounds"
    },
    {
      "id": "07",
      "name": "Lubricant formulation",
      "family": "ML",
      "objective": "weighted formulation quality (property/constraint terms)",
      "constraints": "valid formulation schema, bounds and required composition"
    },
    {
      "id": "08",
      "name": "Material-property prediction",
      "family": "ML",
      "objective": "weighted material-property prediction quality",
      "constraints": "complete test coverage and finite predictions"
    },
    {
      "id": "09",
      "name": "Nuclear-site scheduling",
      "family": "CO",
      "objective": "weighted construction schedule score",
      "constraints": "schema, precedence/resource feasibility and horizon limits"
    },
    {
      "id": "10",
      "name": "Retail stock replenishment",
      "family": "CO",
      "objective": "fulfillment/lateness/cost absolute score",
      "constraints": "complete orders and feasible inventory/transfer records"
    },
    {
      "id": "11",
      "name": "Retail-zone routing",
      "family": "CO",
      "objective": "partition plus routing composite quality",
      "constraints": "all zones/items covered and routing/partition checks pass"
    },
    {
      "id": "12",
      "name": "Steel slitting and cutting",
      "family": "CO",
      "objective": "cut utilization and cut-count score",
      "constraints": "valid cut plan, demand coverage and capacity checks"
    },
    {
      "id": "13",
      "name": "Vessel stowage planning",
      "family": "CO",
      "objective": "weighted vessel-stowage objective components",
      "constraints": "schema, placement, capacity and stability checks"
    },
    {
      "id": "14",
      "name": "Warehouse stock levels",
      "family": "CO",
      "objective": "mean daily policy score minus variability penalty",
      "constraints": "valid daily policy schema and simulator checks"
    },
    {
      "id": "15",
      "name": "Truck-load planning",
      "family": "CO",
      "objective": "0.60 × coverage + 0.25 × length utilization + 0.15 × balance",
      "constraints": "all items assigned once and truck/capacity checks"
    },
    {
      "id": "16",
      "name": "Multi-table APS scheduling",
      "family": "CO",
      "objective": "delay/utilization absolute schedule score",
      "constraints": "all jobs assigned and precedence/resource constraints"
    },
    {
      "id": "17",
      "name": "Cold-rolling sequencing",
      "family": "CO",
      "objective": "coil-sequencing penalty",
      "constraints": "complete sequence, no duplicates and transition feasibility"
    },
    {
      "id": "18",
      "name": "Multi-batch scheduling",
      "family": "CO",
      "objective": "multi-batch scheduling composite score",
      "constraints": "schema, assignment, precedence and capacity checks"
    },
    {
      "id": "19",
      "name": "Multi-line scheduling",
      "family": "CO",
      "objective": "total production penalty",
      "constraints": "complete schedule and line/time feasibility"
    },
    {
      "id": "20",
      "name": "Multi-warehouse shipping",
      "family": "CO",
      "objective": "lexicographic shipping score with tie-break cost",
      "constraints": "all orders served and inventory/route constraints"
    },
    {
      "id": "21",
      "name": "Soybean-meal rescheduling",
      "family": "CO",
      "objective": "weighted smoothness/safety/balance rescheduling score",
      "constraints": "valid schedule and safety/quality checks"
    },
    {
      "id": "22",
      "name": "Wheel-hub scheduling",
      "family": "CO",
      "objective": "total weighted delay",
      "constraints": "two-stage schedule complete and all hard constraints"
    },
    {
      "id": "23",
      "name": "Auction-picking routes",
      "family": "CO",
      "objective": "picking-route distance",
      "constraints": "all required picks served with valid route"
    },
    {
      "id": "24",
      "name": "Quay-crane scheduling",
      "family": "CO",
      "objective": "100,000,000 × number of cranes + total move seconds",
      "constraints": "all hard quay-crane constraints C1–C7"
    },
    {
      "id": "25",
      "name": "Container loading",
      "family": "CO",
      "objective": "container stability/loading quality",
      "constraints": "all items placed once; weight/geometry/stability checks"
    },
    {
      "id": "26",
      "name": "Three-SKU packing",
      "family": "CO",
      "objective": "validated ITEM_C count/value",
      "constraints": "valid container schema and packing/weight checks"
    },
    {
      "id": "27",
      "name": "Flight-crew rostering",
      "family": "CO",
      "objective": "crew-roster combined score",
      "constraints": "complete roster and labor/rest/qualification rules"
    },
    {
      "id": "28",
      "name": "Elevator-call forecasting",
      "family": "ML",
      "objective": "100 / (1 + MAE) minus skew penalty",
      "constraints": "complete calls, finite nonnegative predictions"
    },
    {
      "id": "29",
      "name": "Hydropower dispatch",
      "family": "CO",
      "objective": "total cascade-hydropower energy",
      "constraints": "complete dispatch, bounds and cascade constraints"
    },
    {
      "id": "30",
      "name": "Syngas composition",
      "family": "ML",
      "objective": "Mean R² across syngas targets",
      "constraints": "complete target coverage and finite predictions"
    },
    {
      "id": "31",
      "name": "Credit-default scoring",
      "family": "ML",
      "objective": "0.5 × AUC + 0.5 × KS",
      "constraints": "complete labels/probabilities and valid score range"
    },
    {
      "id": "32",
      "name": "Grid-load forecasting",
      "family": "ML",
      "objective": "MAPE",
      "constraints": "complete time horizon and finite forecasts"
    },
    {
      "id": "33",
      "name": "LLM-traffic forecasting",
      "family": "ML",
      "objective": "traffic RMSE",
      "constraints": "complete horizon and finite forecasts"
    },
    {
      "id": "34",
      "name": "Jewelry-sales forecasting",
      "family": "ML",
      "objective": "0.5 × quantity score + 0.5 × F1",
      "constraints": "complete SKU coverage and valid nonnegative forecasts"
    },
    {
      "id": "35",
      "name": "Corn-futures forecasting",
      "family": "ML",
      "objective": "mean forecasting error",
      "constraints": "complete horizon and finite forecasts"
    },
    {
      "id": "36",
      "name": "Fermentation outcomes",
      "family": "ML",
      "objective": "fermentation MAE",
      "constraints": "complete targets and finite predictions"
    },
    {
      "id": "37",
      "name": "MicroGC pillar design",
      "family": "CO",
      "objective": "0.6 × s_nrmse + 0.2 × s_press + 0.2 × s_perim",
      "constraints": "valid geometry/field outputs and CFD checks"
    },
    {
      "id": "38",
      "name": "Display-fab scheduling",
      "family": "CO",
      "objective": "60 × p_ach + 30 × o_ach + 10 × q_score",
      "constraints": "complete schedule and machine/order constraints"
    },
    {
      "id": "39",
      "name": "Warehouse forklift scheduling",
      "family": "CO",
      "objective": "nested warehouse objective",
      "constraints": "assignment, stack/forklift and temporal constraints"
    },
    {
      "id": "40",
      "name": "Frailty-risk grouping",
      "family": "ML",
      "objective": "W_FIT × R² + W_RISK × (c − 0.5)",
      "constraints": "complete cohort outputs and valid risk/fit fields"
    },
    {
      "id": "41",
      "name": "Casting thermal fields",
      "family": "ML",
      "objective": "mean thermal-field RMSE",
      "constraints": "complete field output and finite values"
    },
    {
      "id": "42",
      "name": "Superalloy strength prediction",
      "family": "ML",
      "objective": "superalloy prediction error",
      "constraints": "complete target coverage and finite values"
    },
    {
      "id": "43",
      "name": "Coagulant dosing",
      "family": "ML",
      "objective": "water-coagulant dosing error",
      "constraints": "complete horizon and finite values"
    },
    {
      "id": "44",
      "name": "Bank-branch siting scores",
      "family": "ML",
      "objective": "branch-siting score MAE",
      "constraints": "all branches scored with valid predictions"
    },
    {
      "id": "45",
      "name": "Data-center rack allocation",
      "family": "CO",
      "objective": "rack-allocation total cost",
      "constraints": "all racks assigned and capacity/power constraints"
    },
    {
      "id": "46",
      "name": "Routing with 3D loading",
      "family": "CO",
      "objective": "100,000 × number of vehicles + routing distance",
      "constraints": "packing and route feasibility"
    },
    {
      "id": "47",
      "name": "Regional fleet dispatch",
      "family": "CO",
      "objective": "regional dispatch total objective",
      "constraints": "all orders assigned; violations penalized in objective"
    },
    {
      "id": "48",
      "name": "Hybrid-energy dispatch",
      "family": "CO",
      "objective": "8,760-hour total income",
      "constraints": "complete dispatch and operational constraints"
    },
    {
      "id": "49",
      "name": "Fuel-tanker routing",
      "family": "CO",
      "objective": "multi-compartment tanker distance",
      "constraints": "all demands served and vehicle/compartment constraints"
    }
  ],
  "authors": [
    {
      "name": "Huaiming Li",
      "affiliation": 1,
      "mark": "*"
    },
    {
      "name": "Can Huang",
      "affiliation": 2,
      "mark": "*"
    },
    {
      "name": "Chufan Wu",
      "affiliation": 1,
      "mark": ""
    },
    {
      "name": "Danyu Liu",
      "affiliation": 1,
      "mark": ""
    },
    {
      "name": "Xiaomin Yuan",
      "affiliation": 1,
      "mark": ""
    },
    {
      "name": "Qinze Li",
      "affiliation": 1,
      "mark": ""
    },
    {
      "name": "Yui Lo",
      "affiliation": 1,
      "mark": ""
    },
    {
      "name": "Wenhui Bai",
      "affiliation": 1,
      "mark": ""
    },
    {
      "name": "Qianlong Wang",
      "affiliation": 1,
      "mark": ""
    },
    {
      "name": "Zengle Ge",
      "affiliation": 1,
      "mark": ""
    },
    {
      "name": "Xiyu Yan",
      "affiliation": 1,
      "mark": ""
    },
    {
      "name": "Yuheng Cheng",
      "affiliation": 3,
      "mark": ""
    },
    {
      "name": "Jinhui Ren",
      "affiliation": 4,
      "mark": ""
    },
    {
      "name": "Guokai Chen",
      "affiliation": 4,
      "mark": ""
    },
    {
      "name": "Jiaqun Liu",
      "affiliation": 5,
      "mark": ""
    },
    {
      "name": "Haoran Li",
      "affiliation": 6,
      "mark": ""
    },
    {
      "name": "Qianhui Liu",
      "affiliation": 7,
      "mark": ""
    },
    {
      "name": "Annan Li",
      "affiliation": 1,
      "mark": ""
    },
    {
      "name": "Jianmin Wu",
      "affiliation": 1,
      "mark": ""
    },
    {
      "name": "Dawei Yin",
      "affiliation": 1,
      "mark": ""
    },
    {
      "name": "Dou Shen",
      "affiliation": 1,
      "mark": "†"
    }
  ],
  "affiliations": [
    {
      "id": 1,
      "name": "FM Agent Team, Baidu AI Cloud"
    },
    {
      "id": 2,
      "name": "Beijing University of Posts and Telecommunications"
    },
    {
      "id": 3,
      "name": "The Chinese University of Hong Kong"
    },
    {
      "id": 4,
      "name": "Tsinghua University"
    },
    {
      "id": 5,
      "name": "Peking University"
    },
    {
      "id": 6,
      "name": "University of the Chinese Academy of Sciences"
    },
    {
      "id": 7,
      "name": "Institute of Automation, Chinese Academy of Sciences"
    }
  ]
};

// Additional paper-backed data for the project page.
// Existing result, case, and author records above are unchanged.
Object.assign(window.FDE_DATA, {
  "ablation": {
    "source": "Paper Table 2; 5_experiments.tex:194–257",
    "cases": 20,
    "repeats": 3,
    "metric": "Case-macro Quality",
    "fdeReference": 1.0,
    "note": "Three conditions on the same 20 cases and eight configurations. Interact-Req reuses the corresponding main runs. Printed deltas use unrounded estimates and may differ slightly from differences of the displayed scores.",
    "conditions": [
      {
        "id": "hidden",
        "label": "Hidden",
        "description": "Initial request and business data; no customer answers."
      },
      {
        "id": "interact",
        "label": "Interact-Req",
        "description": "Initial request and data; clarification required before implementation."
      },
      {
        "id": "full",
        "label": "Full",
        "description": "Complete business information provided up front; no interaction."
      }
    ],
    "results": [
      {
        "scaffold": "opencode",
        "model": "GPT-6-Astra",
        "hidden": 0.1803,
        "interact": 0.58,
        "full": 0.7461,
        "deltaRequired": 0.3997,
        "deltaRemaining": 0.1662
      },
      {
        "scaffold": "opencode",
        "model": "DeepSeek-V4-Pro",
        "hidden": 0.1423,
        "interact": 0.5063,
        "full": 0.6821,
        "deltaRequired": 0.364,
        "deltaRemaining": 0.1758
      },
      {
        "scaffold": "opencode",
        "model": "Gemini-3.1-Pro",
        "hidden": 0.1218,
        "interact": 0.2912,
        "full": 0.4886,
        "deltaRequired": 0.1694,
        "deltaRemaining": 0.1974
      },
      {
        "scaffold": "opencode",
        "model": "Kimi-K3",
        "hidden": 0.1244,
        "interact": 0.3694,
        "full": 0.4488,
        "deltaRequired": 0.245,
        "deltaRemaining": 0.0794
      },
      {
        "scaffold": "codex",
        "model": "GPT-6-Astra",
        "hidden": 0.198,
        "interact": 0.598,
        "full": 0.7946,
        "deltaRequired": 0.4,
        "deltaRemaining": 0.1966
      },
      {
        "scaffold": "deepseek-harness",
        "model": "DeepSeek-V4-Pro",
        "hidden": 0.1998,
        "interact": 0.5642,
        "full": 0.7339,
        "deltaRequired": 0.3644,
        "deltaRemaining": 0.1697
      },
      {
        "scaffold": "gemini",
        "model": "Gemini-3.1-Pro",
        "hidden": 0.1231,
        "interact": 0.3763,
        "full": 0.5184,
        "deltaRequired": 0.2532,
        "deltaRemaining": 0.142
      },
      {
        "scaffold": "kimi",
        "model": "Kimi-K3",
        "hidden": 0.1498,
        "interact": 0.3693,
        "full": 0.7184,
        "deltaRequired": 0.2195,
        "deltaRemaining": 0.349
      }
    ]
  },
  "oracle": {
    "source": "Paper Figure 5 and Appendix Table “Oracle Quality by answer coverage”; 8_appendix.tex:741–783",
    "scaffold": "codex",
    "model": "GPT-6-Astra",
    "repeats": 3,
    "coverage": [
      0,
      25,
      50,
      75,
      100
    ],
    "mean": [
      0.0587,
      0.1575,
      0.3518,
      0.4648,
      0.869
    ],
    "note": "Nested subsets of registered answers, without interaction. At 100%, all registered answers are supplied; this is not the complete business-information file used by Full. Values are means across three repeats, and case trends need not be monotonic.",
    "cases": [
      {
        "id": "09",
        "name": "Nuclear-site scheduling",
        "mean": [
          0.0,
          0.0,
          0.3272,
          0.3206,
          0.977
        ],
        "sd": [
          0.0,
          0.0,
          0.5667,
          0.5553,
          0.0109
        ]
      },
      {
        "id": "11",
        "name": "Retail-zone routing",
        "mean": [
          0.0,
          0.4661,
          0.7442,
          0.7935,
          0.8558
        ],
        "sd": [
          0.0,
          0.4037,
          0.0629,
          0.067,
          0.0582
        ]
      },
      {
        "id": "17",
        "name": "Cold-rolling sequencing",
        "mean": [
          0.2347,
          0.1637,
          0.3357,
          0.426,
          0.6782
        ],
        "sd": [
          0.1189,
          0.0399,
          0.1595,
          0.1418,
          0.2787
        ]
      },
      {
        "id": "38",
        "name": "Display-fab scheduling",
        "mean": [
          0.0,
          0.0,
          0.0,
          0.319,
          0.965
        ],
        "sd": [
          0.0,
          0.0,
          0.0,
          0.5525,
          0.0093
        ]
      }
    ]
  },
  "caseSlugs": {
    "01": "001_car_body_aerodynamic_forecast",
    "02": "002_city_delivery_route_planning",
    "03": "003_crop_yield_sparse_sampling",
    "04": "004_dayahead_power_price_forecast",
    "05": "005_flight_seat_booking_forecast",
    "06": "006_ion_beam_uniformity_optimization",
    "07": "007_lube_oil_inverse_formulation",
    "08": "008_material_property_forward_prediction",
    "09": "009_nuclear_construction_scheduling",
    "10": "010_retail_replenishment_transfer",
    "11": "011_retail_zone_partition_routing",
    "12": "012_silicon_steel_slitting_cutting",
    "13": "013_vessel_stowage_planning",
    "14": "014_warehouse_multi_shuttle_water_level_policy",
    "15": "015_truck_load_planning",
    "16": "016_aps_multi_table_scheduling",
    "17": "017_cold_rolling_coil_sequencing",
    "18": "018_multi_batch_mip_scheduling",
    "19": "019_multi_line_production_scheduling",
    "20": "020_multi_warehouse_shipping_3097",
    "21": "021_soybean_meal_rescheduling",
    "22": "022_wheel_hub_two_stage_scheduling",
    "23": "023_auction_picking_walk",
    "24": "024_quay_crane_60ships",
    "25": "025_container_loading_466",
    "26": "026_container_packing_3sku",
    "27": "027_crew_rostering",
    "28": "028_elevator_call_forecast",
    "29": "029_hydropower_cascade_dispatch",
    "30": "030_biomass_syngas_prediction",
    "31": "031_credit_default_scorecard",
    "32": "032_grid_load_15min_forecast",
    "33": "033_llm_serving_traffic_forecast",
    "34": "034_luxury_boutique_sku_sales",
    "35": "035_corn_futures_price_forecast",
    "36": "036_soy_peptide_fermentation",
    "37": "037_microgc_pillar_cfd_design",
    "38": "038_tftlcd_fab_scheduling",
    "39": "039_warehouse_stack_forklift",
    "40": "040_frailty_cohort_analysis",
    "41": "041_casting_thermal_pinn",
    "42": "042_superalloy_yield_strength",
    "43": "043_water_coagulant_dosing",
    "44": "044_bank_branch_siting_score",
    "45": "045_datacenter_rack_allocation",
    "46": "046_3lcvrp_packing_routing",
    "47": "047_regional_vehicle_order_dispatch",
    "48": "048_energy_dispatch_8760h",
    "49": "049_fuel_tanker_multi_compartment_vrp"
  },
  "caseStudy": {
    "source": "Paper Section 5.3 and Appendix case-10 evidence chain; 8_appendix.tex:828–900",
    "id": "10",
    "name": "Retail stock replenishment",
    "scaffold": "opencode",
    "model": "GPT-6-Astra",
    "condition": "Interact-Req",
    "repeat": 1,
    "days": 17,
    "stores": 73,
    "items": 3055,
    "quality": 0,
    "localWarehouseUnits": 327,
    "affectedStores": 8,
    "violations": 347,
    "questionUnits": 47,
    "clarificationRounds": 3,
    "instructions": 15987,
    "trace": [
      {
        "stage": "Customer requirement",
        "title": "Use internal stock before buying more",
        "detail": "Same-city inventory must be exhausted before supplier purchases. Stockout avoidance takes priority over timeliness and cost."
      },
      {
        "stage": "Clarification",
        "title": "The agent asks about source precedence",
        "detail": "It asks whether same-city stores and the local warehouse take priority over cross-city sources and suppliers. The exact daily inventory transition order is not covered by its questions."
      },
      {
        "stage": "Local implementation",
        "title": "Its own validator reports pass",
        "detail": "The submitted plan contains 15,987 instructions: 15,515 transfers and 472 purchases. Plan metadata states the daily transition order, but a declaration does not enforce the rule."
      },
      {
        "stage": "Hidden evaluation",
        "title": "327 local units remain when it buys from a supplier",
        "detail": "The evaluator identifies same-city inventory violations at eight stores and 347 constraint violations in total. All five extractor/evaluator routes mark the plan invalid: Quality = 0."
      }
    ],
    "note": "A measured run from the paper, not a simulated website conversation. The trace illustrates the gap between asking about a requirement and enforcing it in a final artifact."
  },
  "audits": {
    "source": "Paper Section 4.3 and Appendix audits; 4_protocol.tex:140–159; 8_appendix.tex:904–1020",
    "experts": 2,
    "configurations": 18,
    "customerAnswers": {
      "sample": 1303,
      "supportedA": 95.93,
      "supportedB": 90.48,
      "agreement": 92.1,
      "kappa": 0.385
    },
    "artifactExtractions": {
      "sample": 120,
      "faithfulA": 93.33,
      "faithfulB": 91.0,
      "agreement": 91.67,
      "kappa": 0.7656,
      "note": "Fidelity percentages are among assessable extractions in a sample enriched for extraction issues. Kappa measures agreement between expert labels, not accuracy against adjudicated ground truth."
    },
    "clarificationItems": {
      "sample": 120,
      "panelAgreementA": 94.74,
      "panelAgreementB": 93.33,
      "panelKappaA": 0.8947,
      "panelKappaB": 0.8667,
      "note": "Comparisons exclude uncertain labels; the comparison with expert A contains 114 items."
    },
    "note": "These checks support reliability on the audited subsets, but do not eliminate simulator or extraction risk."
  },
  "zeroScores": {
    "source": "Paper Appendix zero-score analysis; 8_appendix.tex:784–827",
    "totalRuns": 2646,
    "zeroRuns": 548,
    "invalidRuns": 509,
    "validZeroRuns": 39,
    "signatures": [
      {
        "name": "Constraint or feasibility violation",
        "count": 430,
        "share": 78.5
      },
      {
        "name": "Fatal case-specific check",
        "count": 29,
        "share": 5.3
      },
      {
        "name": "Schema or output validation error",
        "count": 8,
        "share": 1.5
      },
      {
        "name": "Evaluator or scoring exception",
        "count": 4,
        "share": 0.7
      },
      {
        "name": "Quality or threshold warning",
        "count": 16,
        "share": 2.9
      },
      {
        "name": "No explicit evaluator message",
        "count": 61,
        "share": 11.1
      }
    ]
  }
});
