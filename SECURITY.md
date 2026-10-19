# Security policy

## Supported versions

Only the `main` branch gets security fixes. Older commits and images do not.

## Report a vulnerability

Do not open a public issue for a security problem.

1. Go to the [Security tab](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/security) of this repo.
2. Click **Report a vulnerability**. This opens a private report that only the maintainer can see.
3. Tell us the problem, the steps to see it, the files and the commit. Do not include real keys or tokens.

If the report form is not there, use the contact details on the GitHub profile of the maintainer, [@Elevatormusic](https://github.com/Elevatormusic).

The maintainer tries to reply in 7 days. You get a message when a fix is ready. You get credit if you want it.

## What is in scope

- The scripts in this repo: `start.sh`, `stop.sh`, `status.sh`, `verify.sh`, `scripts/`, `docker/`, `tests/`.
- Default settings that expose the server or the nodes. For example, a default that serves the API on the LAN with no API key.
- A secret or a machine-specific value in a committed file.

Problems in the engine ([TensorFold](https://github.com/Elevatormusic/TensorFold)), vLLM, the model or the NVIDIA software go to those projects. If you are not sure, report it here and we send it on.

## Safe use

- Keep `BIND_HOST=127.0.0.1` unless you need network access. If you serve the LAN, set `API_KEY_FILE` ([docs/CONNECT.md](docs/CONNECT.md)).
- Do not commit `.env`, keys or tokens. [CONTRIBUTING.md](CONTRIBUTING.md#no-secrets-and-no-machine-specific-values) gives the rules.
