# DEPLOYMENT PROCESS

use make release command:
it creates tag, pushes to github and makes release that triggers deployment process
release environment determined by suffix format `(<env_name>)` like `release v0.11 (prod)`

list of environments supported is at release.yaml file

to use gommand you have to install gh command locally, and login with it

`./make_release.sh <VERSION> <ENV_NAME> [OPTIONAL DESCRIPTION]`

example:
`./make_release.sh 0.11 prod 'some description'`

