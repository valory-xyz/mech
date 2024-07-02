#!/bin/bash

export VERSION=$1
export ENV=$2
export DESCRIPTION=$3

if [ -z $VERSION ]; then
    echo no version
    exit 1
fi

if [ -z $ENV ]; then
    echo no env
    exit 1
fi

export TAG_NAME=release_${VERSION}_${ENV}

echo make tag $TAG_NAME ...
git tag $TAG_NAME
echo push tag $TAG_NAME ...
git push origin $TAG_NAME
echo create release $TAG_NAME ...

if gh release create $TAG_NAME --title "Release $VERSION ($ENV)" --notes "$DESCRIPTION"; then
    echo done
else
    echo Error!
fi
