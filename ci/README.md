# About

This folder contains files used for automatic testing in the [CI](https://en.wikipedia.org/wiki/Continuous_integration).

Please read the [Vikunja CI documentation](https://vikunja.readthedocs.io/en/latest/development/ci.html).

# Structure

* **job_generator:** Python scripts to generate the actual CI test jobs.
* **custom_jobs:** Yaml files with user-defined jobs that differ significantly from the jobs created with the job generator.
* **gitlab_scripts:** This folder contains all scripts, which are executed in the gitlab jobs.
* **check_cpp_code_style.sh:** This script is applied to the source code to check, that are all clang-format rules are fulfilled.
