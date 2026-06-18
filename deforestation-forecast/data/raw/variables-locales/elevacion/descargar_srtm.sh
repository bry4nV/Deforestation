#!/bin/bash

GREP_OPTIONS=''

cookiejar=$(mktemp cookies.XXXXXXXXXX)
netrc=$(mktemp netrc.XXXXXXXXXX)
chmod 0600 "$cookiejar" "$netrc"
function finish {
  rm -rf "$cookiejar" "$netrc"
}

trap finish EXIT
WGETRC="$wgetrc"

prompt_credentials() {
    echo "Enter your Earthdata Login or other provider supplied credentials"
    read -p "Username (bry4nv): " username
    username=${username:-bry4nv}
    read -s -p "Password: " password
    echo "machine urs.earthdata.nasa.gov login $username password $password" >> $netrc
    echo
}

exit_with_error() {
    echo
    echo "Unable to Retrieve Data"
    echo
    echo $1
    echo
    echo "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W078.SRTMGL1.hgt/S04W078.SRTMGL1.hgt.zip"
    echo
    exit 1
}

prompt_credentials
  detect_app_approval() {
    approved=`curl -s -b "$cookiejar" -c "$cookiejar" -L --max-redirs 5 --netrc-file "$netrc" https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W078.SRTMGL1.hgt/S04W078.SRTMGL1.hgt.zip -w '\n%{http_code}' | tail  -1`
    if [ "$approved" -ne "200" ] && [ "$approved" -ne "301" ] && [ "$approved" -ne "302" ]; then
        # User didn't approve the app. Direct users to approve the app in URS
        exit_with_error "Please ensure that you have authorized the remote application by visiting the link below "
    fi
}

setup_auth_curl() {
    # Firstly, check if it require URS authentication
    status=$(curl -s -z "$(date)" -w '\n%{http_code}' https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W078.SRTMGL1.hgt/S04W078.SRTMGL1.hgt.zip | tail -1)
    if [[ "$status" -ne "200" && "$status" -ne "304" ]]; then
        # URS authentication is required. Now further check if the application/remote service is approved.
        detect_app_approval
    fi
}

setup_auth_wget() {
    # The safest way to auth via curl is netrc. Note: there's no checking or feedback
    # if login is unsuccessful
    touch ~/.netrc
    chmod 0600 ~/.netrc
    credentials=$(grep 'machine urs.earthdata.nasa.gov' ~/.netrc)
    if [ -z "$credentials" ]; then
        cat "$netrc" >> ~/.netrc
    fi
}

fetch_urls() {
  if command -v curl >/dev/null 2>&1; then
      setup_auth_curl
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        curl -f -b "$cookiejar" -c "$cookiejar" -L --netrc-file "$netrc" -g -o $stripped_query_params -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  elif command -v wget >/dev/null 2>&1; then
      # We can't use wget to poke provider server to get info whether or not URS was integrated without download at least one of the files.
      echo
      echo "WARNING: Can't find curl, use wget instead."
      echo "WARNING: Script may not correctly identify Earthdata Login integrations."
      echo
      setup_auth_wget
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        wget --load-cookies "$cookiejar" --save-cookies "$cookiejar" --output-document $stripped_query_params --keep-session-cookies -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  else
      exit_with_error "Error: Could not find a command-line downloader.  Please install curl or wget"
  fi
}

fetch_urls <<'EDSCEOF'
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W078.SRTMGL1.hgt/S04W078.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S05W080.SRTMGL1.hgt/S05W080.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S03W070.SRTMGL1.hgt/S03W070.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S09W078.SRTMGL1.hgt/S09W078.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W070.SRTMGL1.hgt/S04W070.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S06W078.SRTMGL1.hgt/S06W078.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S06W074.SRTMGL1.hgt/S06W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S09W080.SRTMGL1.hgt/S09W080.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S07W080.SRTMGL1.hgt/S07W080.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S03W074.SRTMGL1.hgt/S03W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/N00W075.SRTMGL1.hgt/N00W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S02W074.SRTMGL1.hgt/S02W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S05W078.SRTMGL1.hgt/S05W078.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S08W080.SRTMGL1.hgt/S08W080.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S03W078.SRTMGL1.hgt/S03W078.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S07W074.SRTMGL1.hgt/S07W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S05W074.SRTMGL1.hgt/S05W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S08W074.SRTMGL1.hgt/S08W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W080.SRTMGL1.hgt/S04W080.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S01W074.SRTMGL1.hgt/S01W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S07W078.SRTMGL1.hgt/S07W078.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W074.SRTMGL1.hgt/S04W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S08W078.SRTMGL1.hgt/S08W078.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S06W080.SRTMGL1.hgt/S06W080.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S05W070.SRTMGL1.hgt/S05W070.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S02W071.SRTMGL1.hgt/S02W071.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S06W075.SRTMGL1.hgt/S06W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S03W079.SRTMGL1.hgt/S03W079.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S07W075.SRTMGL1.hgt/S07W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S05W075.SRTMGL1.hgt/S05W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S05W071.SRTMGL1.hgt/S05W071.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S03W075.SRTMGL1.hgt/S03W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S03W071.SRTMGL1.hgt/S03W071.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S05W081.SRTMGL1.hgt/S05W081.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S05W079.SRTMGL1.hgt/S05W079.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W071.SRTMGL1.hgt/S04W071.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S02W075.SRTMGL1.hgt/S02W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S09W074.SRTMGL1.hgt/S09W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S07W079.SRTMGL1.hgt/S07W079.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S01W075.SRTMGL1.hgt/S01W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S06W079.SRTMGL1.hgt/S06W079.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W081.SRTMGL1.hgt/S04W081.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W079.SRTMGL1.hgt/S04W079.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S07W081.SRTMGL1.hgt/S07W081.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S06W081.SRTMGL1.hgt/S06W081.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W075.SRTMGL1.hgt/S04W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W077.SRTMGL1.hgt/S04W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S03W072.SRTMGL1.hgt/S03W072.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W072.SRTMGL1.hgt/S04W072.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S06W082.SRTMGL1.hgt/S06W082.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S08W076.SRTMGL1.hgt/S08W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W076.SRTMGL1.hgt/S04W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S09W076.SRTMGL1.hgt/S09W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S07W076.SRTMGL1.hgt/S07W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S07W082.SRTMGL1.hgt/S07W082.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S05W076.SRTMGL1.hgt/S05W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/N00W076.SRTMGL1.hgt/N00W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S05W082.SRTMGL1.hgt/S05W082.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S06W072.SRTMGL1.hgt/S06W072.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S06W076.SRTMGL1.hgt/S06W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S03W076.SRTMGL1.hgt/S03W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S02W076.SRTMGL1.hgt/S02W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S08W079.SRTMGL1.hgt/S08W079.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S08W075.SRTMGL1.hgt/S08W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S05W072.SRTMGL1.hgt/S05W072.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S02W072.SRTMGL1.hgt/S02W072.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S09W079.SRTMGL1.hgt/S09W079.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S09W075.SRTMGL1.hgt/S09W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S01W076.SRTMGL1.hgt/S01W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S10W079.SRTMGL1.hgt/S10W079.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S05W073.SRTMGL1.hgt/S05W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S09W077.SRTMGL1.hgt/S09W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S05W077.SRTMGL1.hgt/S05W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S08W077.SRTMGL1.hgt/S08W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S06W077.SRTMGL1.hgt/S06W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S02W073.SRTMGL1.hgt/S02W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S09W073.SRTMGL1.hgt/S09W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S07W077.SRTMGL1.hgt/S07W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S02W077.SRTMGL1.hgt/S02W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S10W071.SRTMGL1.hgt/S10W071.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S11W075.SRTMGL1.hgt/S11W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S03W073.SRTMGL1.hgt/S03W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S10W075.SRTMGL1.hgt/S10W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S01W077.SRTMGL1.hgt/S01W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S04W073.SRTMGL1.hgt/S04W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S06W073.SRTMGL1.hgt/S06W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S03W077.SRTMGL1.hgt/S03W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S07W073.SRTMGL1.hgt/S07W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S15W071.SRTMGL1.hgt/S15W071.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S18W069.SRTMGL1.hgt/S18W069.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S11W079.SRTMGL1.hgt/S11W079.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S11W072.SRTMGL1.hgt/S11W072.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S13W072.SRTMGL1.hgt/S13W072.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S17W075.SRTMGL1.hgt/S17W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S10W076.SRTMGL1.hgt/S10W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S12W071.SRTMGL1.hgt/S12W071.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S14W075.SRTMGL1.hgt/S14W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S17W069.SRTMGL1.hgt/S17W069.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S11W071.SRTMGL1.hgt/S11W071.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S19W071.SRTMGL1.hgt/S19W071.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S12W069.SRTMGL1.hgt/S12W069.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S13W069.SRTMGL1.hgt/S13W069.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S13W075.SRTMGL1.hgt/S13W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S15W069.SRTMGL1.hgt/S15W069.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S14W071.SRTMGL1.hgt/S14W071.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S14W069.SRTMGL1.hgt/S14W069.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S16W075.SRTMGL1.hgt/S16W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S12W072.SRTMGL1.hgt/S12W072.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S18W071.SRTMGL1.hgt/S18W071.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S12W075.SRTMGL1.hgt/S12W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S10W073.SRTMGL1.hgt/S10W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S13W073.SRTMGL1.hgt/S13W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S16W073.SRTMGL1.hgt/S16W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S13W071.SRTMGL1.hgt/S13W071.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S12W076.SRTMGL1.hgt/S12W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S11W076.SRTMGL1.hgt/S11W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S16W071.SRTMGL1.hgt/S16W071.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S17W071.SRTMGL1.hgt/S17W071.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S16W069.SRTMGL1.hgt/S16W069.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S10W072.SRTMGL1.hgt/S10W072.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S15W075.SRTMGL1.hgt/S15W075.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S18W072.SRTMGL1.hgt/S18W072.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S13W077.SRTMGL1.hgt/S13W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S12W077.SRTMGL1.hgt/S12W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S14W077.SRTMGL1.hgt/S14W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S14W072.SRTMGL1.hgt/S14W072.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S12W073.SRTMGL1.hgt/S12W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S14W073.SRTMGL1.hgt/S14W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S15W073.SRTMGL1.hgt/S15W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S17W073.SRTMGL1.hgt/S17W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S11W077.SRTMGL1.hgt/S11W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S11W073.SRTMGL1.hgt/S11W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S17W072.SRTMGL1.hgt/S17W072.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S15W077.SRTMGL1.hgt/S15W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S14W076.SRTMGL1.hgt/S14W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S16W076.SRTMGL1.hgt/S16W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S18W073.SRTMGL1.hgt/S18W073.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S10W077.SRTMGL1.hgt/S10W077.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S15W076.SRTMGL1.hgt/S15W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S13W076.SRTMGL1.hgt/S13W076.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S15W072.SRTMGL1.hgt/S15W072.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S10W074.SRTMGL1.hgt/S10W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S12W074.SRTMGL1.hgt/S12W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S16W072.SRTMGL1.hgt/S16W072.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S11W074.SRTMGL1.hgt/S11W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S18W070.SRTMGL1.hgt/S18W070.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S15W074.SRTMGL1.hgt/S15W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S16W070.SRTMGL1.hgt/S16W070.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S14W074.SRTMGL1.hgt/S14W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S13W078.SRTMGL1.hgt/S13W078.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S14W070.SRTMGL1.hgt/S14W070.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S10W078.SRTMGL1.hgt/S10W078.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S13W070.SRTMGL1.hgt/S13W070.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S16W074.SRTMGL1.hgt/S16W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S13W074.SRTMGL1.hgt/S13W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S19W070.SRTMGL1.hgt/S19W070.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S17W070.SRTMGL1.hgt/S17W070.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S12W078.SRTMGL1.hgt/S12W078.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S12W070.SRTMGL1.hgt/S12W070.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S11W078.SRTMGL1.hgt/S11W078.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S17W074.SRTMGL1.hgt/S17W074.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S15W070.SRTMGL1.hgt/S15W070.SRTMGL1.hgt.zip
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003/S11W070.SRTMGL1.hgt/S11W070.SRTMGL1.hgt.zip
EDSCEOF