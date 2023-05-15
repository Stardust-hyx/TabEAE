#!/bin/bash
CURDIR=$(pwd)

if [[ "$CURDIR" =~ "TabEAE/data" ]]
then
	echo "please run this script under the root dir of the project, eg directory TabEAE"
	echo "please input the command ' cd .. ' then press return  "
	exit -1
else
	echo "downloading data from a server ... "

fi

# Download RAMS
wget -c https://nlp.jhu.edu/rams/RAMS_1.0b.tar.gz
tar -zxvf ./RAMS_1.0b.tar.gz
rm -rf ./RAMS_1.0b.tar.gz
mv ./RAMS_1.0/data ./data/RAMS_1.0/data
RAMSDIR=./data/RAMS_1.0
mkdir -p $RAMSDIR/data_final
python $RAMSDIR/squeeze_merge.py --indir $RAMSDIR/data --outdir $RAMSDIR/data_final
rm -rf ./RAMS_1.0


# Download WIKIEVENTS
WIKIDIR=./data/WikiEvent
mkdir -p $WIKIDIR/data
mkdir -p $WIKIDIR/data_final
wget -c -P $WIKIDIR/data  https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/train.jsonl
wget -c -P $WIKIDIR/data  https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/dev.jsonl
wget -c -P $WIKIDIR/data  https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/test.jsonl
python $WIKIDIR/split.py --indir $WIKIDIR/data --outdir $WIKIDIR/data_final

# Download MLEE
wget -c http://www.nactem.ac.uk/MLEE/MLEE-1.0.2-rev1.tar.gz
tar -zxvf ./MLEE-1.0.2-rev1.tar.gz
rm -rf ./MLEE-1.0.2-rev1.tar.gz
MLEEDIR=./data/MLEE
SCRIPT=./data/MLEE/scripts
mkdir -p $MLEEDIR/raw
python $SCRIPT/MLEE_to_deepeventmine.py --indir ./MLEE-1.0.2-rev1/standoff/test --outdir $MLEEDIR/raw
python $SCRIPT/deepeventmine_to_TabEAE.py --indir $MLEEDIR/raw --outdir $MLEEDIR/
rm -rf ./MLEE-1.0.2-rev1
