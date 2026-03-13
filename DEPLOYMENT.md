# DEPLOYMENT PROCESS

Use the release script to create a tag, push it, and create a GitHub release.
That release triggers the deployment workflow.

Release environment is determined by the release title suffix `(<env_name>)`, for example: `Release 0.11 (prod)`.
Supported suffixes are:
- `prod` (production deployment)
- `staging` (staging deployment)

You need `gh` installed locally and authenticated before running the script.

`./make_release.sh <VERSION> <ENV_NAME> [OPTIONAL DESCRIPTION]`

example:
`./make_release.sh 0.11 prod 'some description'`
