# Analysis notebooks

The notebooks are execution entry points for the ongoing research workflow;
they are not evidence that the planned full experiments have been completed.

## `01_colab_training.ipynb`

This Colab-oriented notebook:

1. clones the repository and installs the environment;
2. mounts a user-authorized Google Drive folder for checkpoint persistence;
3. downloads the documented ETCI 2021 community mirror after the user reviews
   the official data terms;
4. verifies event discovery and channel construction;
5. runs a short smoke experiment before optional full baseline training;
6. evaluates the resulting checkpoint on the held-out Florence region.

The `/content/` paths are intentionally Colab-specific. Repository notebooks
are stored without cell outputs, checkpoints, credentials, or personal Drive
identifiers. Full training remains optional and results must not be presented as
project findings until the run artifacts and configuration are recorded.
