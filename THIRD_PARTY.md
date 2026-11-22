# Third-party components

| Component | Location | Treatment |
| --- | --- | --- |
| PX4 message definitions | `src/px4_msgs/` | Bundled snapshot; its source and license are preserved |
| PX4 Autopilot, g2o, cg2o, IPOPT (with MUMPS, ASL, SPRAL), XRCE-DDS Agent, QGroundControl | `externals/` after setup | Fetched at pinned commits and built by `scripts/build_externals.sh` ([versions](INSTALL.md#versions)); not part of this repository |
| HSL (MA27, MA57) | `install/externals/hsl` after setup | Licensed separately (free for academic use); the source is supplied by the user (`HSL_SOURCE`) |
| CasADi, NumPy, PyYAML, Matplotlib and build/test tools | Python environment | Installed from requirement files; not vendored |
| Solver tuning tool | `tools/solver_tuning/` | Separate package; see its [notice](tools/solver_tuning/NOTICE.md) |

The exact upstream Git commit of the bundled `px4_msgs` snapshot is not
recorded. Record that revision when deliberately updating the
interfaces, and match the PX4 firmware's message definitions. The bundled
[license](src/px4_msgs/LICENSE) remains authoritative for those files.

External build revisions and options are described in [installation](INSTALL.md).
