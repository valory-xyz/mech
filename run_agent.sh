rm -r agent
find . -empty -type d -delete  # remove empty directories to avoid wrong hashes
autonomy packages lock
autonomy fetch --local --agent eightballer/mech --alias agent && cd agent
cp $PWD/../ethereum_private_key.txt .
autonomy add-key ethereum ethereum_private_key.txt
aea install
autonomy issue-certificates
aea -s run
