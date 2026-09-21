# Security

IMGEN is a **local** studio. It binds to `127.0.0.1` by default because the process can read uploaded images, write under `IMGEN_HOME`, and run GPU inference.

Do not expose `--host 0.0.0.0` on an untrusted network.

Tokens stored in `~/.imgen/config.json` are local secrets. The file is created with mode `600` where the OS allows it. Do not commit that file.

Report vulnerabilities privately via GitHub Security Advisories on the repository, or by opening a confidential issue if advisories are not enabled yet.
