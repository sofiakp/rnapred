#!/bin/bash

usage()
{
cat <<EOF
usage: `basename $0` options BED MARK OUTDIR
Gets the signal on peaks of a mark from Roadmap that overlap a 
given BED file. First, the overlapping regions between the peaks
and the BED file are found. Then, the average signal of the mark
on each of the overlapping regions is computed.

It reads a list of sample names from STDIN (eg. E001, E002, etc). 
It will look for peak files 
<peak_dir>/<sample>-<MARK>.<peak_suf>. 
For each such file, it will look for a signal file in signal_dir,
that starts with <sample>-<MARK>.

It assumes that the BED filename has the form
gencode.v19.annotation.noM.genes.<something>.bed
so it will use <something> as the suffix of the output files.

OPTIONS:
   -h     Show this message and exit
   -p DIR Directory with peaks 
          [default $LABREPO/epigenomeRoadmap/peaks/stdnames30M/combrep/]
   -f STR Suffix of peak files [default narrowPeak.gz]
   -s DIR Signal dir 
          [default $LABREPO/epigenomeRoadmap/signal/stdnames30M/macs2signal/pval/]
EOF
}

PEAKDIR=/srv/gsfs0/projects/kundaje/commonRepository/epigenomeRoadmap/peaks/stdnames30M/combrep/
PEAKSUF=narrowPeak.gz
SIGNALDIR=/srv/gsfs0/projects/kundaje/commonRepository/epigenomeRoadmap/signal/stdnames30M/macs2signal/pval/

while getopts "hp:f:s:" opt
do
    case $opt in
	h)
	    usage; exit;;
	p)
	    PEAKDIR=$OPTARG;;
	f)
	    PEAKSUF=$OPTARG;;
	s)
	    SIGNALDIR=$OPTARG;;
	?)
	    usage
	        exit 1;;
    esac    
done

shift "$((OPTIND - 1))"
if [ $# -ne 3 ]; then 
    usage; exit 1;
fi

BED=$1
MARK=$2
OUTDIR=$3

if [ ! -d ${OUTDIR}/tmp ]; then
    mkdir -p ${OUTDIR}/tmp
fi

OUTSUF=$(basename $BED)
OUTSUF=${OUTSUF/gencode.v19.annotation.noM.genes./}
OUTSUF=${OUTSUF/.bed/}

UCSCTOOLS=${HOME}/software/ucsc

while read sample; do
    script=${OUTDIR}/tmp/${sample}-${MARK}.${OUTSUF}.sh
    tmp_bed=${OUTDIR}/tmp/${sample}-${MARK}.${OUTSUF}.bed
    tmp_out=${OUTDIR}/tmp/${sample}-${MARK}.${OUTSUF}.txt
    errfile=${OUTDIR}/tmp/${sample}-${MARK}.${OUTSUF}.err
    
    outfile=${OUTDIR}/${sample}-${MARK}.${OUTSUF}.bed
    if [ -f $outfile ]; then
	echo "$outfile exists. Skipping" 1>&2
	continue
    fi

    in_bed=${PEAKDIR}/${sample}-${MARK}.${PEAKSUF}
    if [ ! -f ${in_bed} ]; then
	echo "Input bed file $in_bed is missing." 1>&2
	continue
    fi
    if [ `ls $SIGNALDIR | egrep ${sample}-$MARK | wc -l` -ne 1 ]; then
	echo "Signal file for ${sample}-$MARK is missing or ambiguous." 1>&2
	continue
    fi

    signal_file=`ls $SIGNALDIR | egrep ${sample}-$MARK`
    signal_file=${SIGNALDIR}/$signal_file

    echo "#!/bin/bash" > $script
    echo "module add bedtools/2.19.1" >> $script
    # Notice that the intersection doesn't use -wa or -wb, so only
    # the intersecting parts will be in the output. 
    # The sort ensures that overlaps for the same gene will be in 
    # consecutive lines.
    echo "intersectBed -a $BED -b ${in_bed} | sort -k7,7 | awk 'BEGIN{OFS=\"\t\"}{print \$1,\$2,\$3,\$7\"_\"NR}' > $tmp_bed" >> $script
    echo "$UCSCTOOLS/bigWigAverageOverBed -bedOut=$outfile ${signal_file} ${tmp_bed} ${tmp_out}" >> $script
    
    qsub -N ${sample}-${MARK} -q standard -l h_vmem=4G -l h_rt=2:00:00 -e $errfile -o /dev/null $script
done