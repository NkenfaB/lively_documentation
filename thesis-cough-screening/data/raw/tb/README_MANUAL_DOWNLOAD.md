# Manual Download Required: tb

Dataset: Controlled-access TB cough dataset.

This dataset is not downloaded automatically by the project scaffold.

## Why

- Access may require account creation, approval, registration, or usage acceptance.
- The pipeline intentionally avoids faking gated downloads.

## Sources

- Primary source: https://tbdata.ucsf.edu/s/rdc-dataset/a0U5w00000fTCKiEAO/ds000731
- Supporting source: https://www.nature.com/articles/s41597-024-03972-z
- Supporting source: https://www.synapse.org/Synapse:syn31472953

## Required Steps

1. Open the UCSF dataset page and review the dataset access conditions.
2. Create or sign in to a Synapse account if the released files are routed through Synapse.
3. Request access or accept the required usage terms if prompted.
4. Download the approved TB cough audio and accompanying metadata manually.
5. Place the extracted files into data/raw/tb/ while preserving the original metadata files.
6. Re-run the audit and metadata unification scripts after the files are present.

## Expected Result

Place the extracted dataset contents inside `/workspace/lively/data/raw/tb`.
The downstream scripts will detect the files automatically if metadata and audio files are present.
