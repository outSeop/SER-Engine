# SER Simulation Analysis Report

## Scenario Summary Statistics

| scenario             |   ('final_alive_agents', 'mean') |   ('final_alive_agents', 'std') |   ('final_equilibrium_score', 'mean') |   ('final_equilibrium_score', 'std') |   ('mean_replication_rate', 'mean') |   ('mean_replication_rate', 'std') |   ('total_replications', 'mean') |   ('total_replications', 'std') |   ('persistence_time', 'mean') |   ('persistence_time', 'std') |
|:---------------------|---------------------------------:|--------------------------------:|--------------------------------------:|-------------------------------------:|------------------------------------:|-----------------------------------:|---------------------------------:|--------------------------------:|-------------------------------:|------------------------------:|
| NO_OBJECTIVE         |                                0 |                             nan |                                 0.397 |                                  nan |                               0     |                                nan |                            0     |                             nan |                            134 |                           nan |
| PROGRAMMED_OBJECTIVE |                               14 |                             nan |                                 0.808 |                                  nan |                               0.004 |                                nan |                            1.912 |                             nan |                            499 |                           nan |
| PURE_RATIONAL        |                               10 |                             nan |                                 1     |                                  nan |                               0     |                                nan |                            0     |                             nan |                            499 |                           nan |
| SELF_PRESERVING      |                                9 |                             nan |                                 1     |                                  nan |                               0     |                                nan |                            0     |                             nan |                            499 |                           nan |
| SELF_REPLICATING     |                               19 |                             nan |                                 1     |                                  nan |                               0.004 |                                nan |                            1.973 |                             nan |                            499 |                           nan |

## SER Hypothesis Interpretation

- **PURE_RATIONAL**: Expected to reach static equilibrium quickly with zero replication.
- **NO_OBJECTIVE**: Expected near-zero activity, immediate stasis.
- **SELF_PRESERVING**: Expected stable but static, no replication.
- **PROGRAMMED_OBJECTIVE**: Expected active but low/no sustained replication.
- **SELF_REPLICATING**: Expected dynamic, persistent replication, lineage divergence.

