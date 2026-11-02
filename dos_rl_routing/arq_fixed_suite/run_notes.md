# Run provenance

The full network runner completed every experiment, figure and report stage with the ARQ-fixed simulator. During the run the outer safety wrapper was edited to strengthen future output/resume guards. Because bash was still reading that wrapper, the outer process reported a shell parsing error *after* the inner runner had completed and written all stage markers and `status.txt` as complete. The wrapper was then invoked with `--resume`; all ten completed stages were verified and skipped, and the invocation exited successfully. No experiment was rerun or mixed during that recovery.

`network_sim.sha256` and `suite_source.sha256` record simulator and suite source fingerprints; `software_versions.txt` records the runtime environment. The existing `corrected_suite/` CSVs are separate historical results with the old ARQ error. E1 and E8 were not part of this network-only suite.
