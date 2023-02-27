#!/bin/bash

jq -r '. | select(.repository != null) | "\(.repository)/HEAD/\(.commit)/\(.path)"' |
awk -F '/' '{
	if ( $1 == "gitlab.com" ){
		printf "https://"$1"/"; 
		for (j=2; j<NF;j++) {
			if ($j == "HEAD"){
				break;
			}
			printf $j "/";
		}
		printf "-/raw/";
		for (i=j+1; i<NF; i++) printf $i "/"; print $NF;
	}
	else if ( $1 == "github.com" ) {
		printf "https://raw.githubusercontent.com/"; 
		for (i=2; i<NF; i++) printf $i "/"; print $NF;
	}
}' |
sed -e "s/\/HEAD//g"
#cut -d '/' -f 2- 
#awk '{print "https://raw.githubusercontent.com/"$1}'
#jq -r '. | select(.repository != null) | "\(.repository)/\(.commit)/\(.path)"'  |
#cut -d '/' -f 2- |
#awk '{print "https://raw.githubusercontent.com/"$1}'
	#if ( $1 == "gitlab.com" )
	#	print "https://$1$2$3/-/raw/";
#https://gitlab.com/cryptsetup/cryptsetup/-/raw/1a55b69a0f8028150a9c93455f24617bc7c8bd61/po/de.po