# Metrics reference

## Physics check (`physics_results.csv`)

| Column | Type | Description |
|---|---|---|
| `asset_id` | str | Object identifier |
| `collision_mode` | str | `convex_hull`, `vhacd`, or `raw` |
| `physics_settles` | bool | Object reached near-zero velocity before timeout |
| `physics_stable` | bool | Settles + no fly-away + no floor penetration |
| `displacement_m` | float | Final XY displacement from spawn position (m) |
| `flies_away` | bool | XY displacement > 1.5 m |
| `penetration_y_m` | float | Minimum Z of object AABB (negative = below floor) |
| `floor_penetration` | bool | Penetration below −0.05 m |
| `settle_time_s` | float | Time to settle (s), null if never settled |
| `wall_time_s` | float | Elapsed wall-clock time per asset (s) |
| `contact_points_at_rest` | int | Number of floor contact points at rest |
| `error` | str | Error message if the check failed, null otherwise |

## Graspability check (`graspability_results.csv`)

| Column | Type | Description |
|---|---|---|
| `asset_id` | str | Object identifier |
| `graspable` | bool | At least one valid antipodal grasp found |
| `grasp_score` | float | Geometric grasp quality score [0, 1] |
| `num_grasps` | int | Number of valid antipodal grasp candidates |
| `error` | str | Error message if the check failed, null otherwise |
