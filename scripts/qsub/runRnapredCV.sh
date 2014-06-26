#!/bin/bash

usage()
{
cat <<EOF
usage: `basename $0` options INDIR OUTDIR FILES
Runs rnapredCV.py for all the files with paths:
<INDIR>/<sample>_feat_mat.npz. The list of samples is read
from a file. The output files are written in
<OUTDIR>/<sample>_cv.npz.

When -g is specified, then all the files in INDIR
that end in _feat_mat.npz will be read and concatenated before
running the cross validation.

OPTIONS:
   -h     Show this message and exit
   -m STR Classification/Regression method. One of rf or logLasso.
          [default: rf]
   -t STR Comma separated list of values for the number of trees.
   -f STR Comma separated list of values for the min number of 
          examples in leaves.
   These two only apply when the method is rf.
   -a STR Comma separated list of values for the alpha parameter
          of Lasso.
   -c STR Comma separated list of values for the C parameter
          of Logistic Regression.
   These two only apply when the method is logLasso.
   -p NUM Number of processors to request [default: 4].
   -g     Merge all the samples.
   -s     Merge but exclude each of the files in the file list.
   -r     Just run, instead of submitting jobs.
   -n     No CV, just return the model trained on all data.
EOF
}

TREES="10"
LEAF="1"
ALPHAS="1"
C="1"
MERGE=0
METHOD="rf"
RUN=0
NPROC=4
NOCV=""
SPLIT=0

while getopts "ht:f:a:c:m:p:grns" opt
do
    case $opt in
	h)
	    usage; exit;;
	t)
	    TREES=$OPTARG;;
	f)
	    LEAF=$OPTARG;;
	a)
	    ALPHAS=$OPTARG;;
	c)
	    C=$OPTARG;;
	p)
	    NPROC=$OPTARG;;
	g)
	    MERGE=1;;
	r)
	    RUN=1;;
	m)
	    METHOD=$OPTARG;;
	n)
	    NOCV="--nocv";;
	s)
	    SPLIT=1;;
	?)
	    usage
	    exit 1;;
    esac    
done

shift "$((OPTIND - 1))"
if [ $# -ne 3 ]; then 
    usage; exit 1;
fi

INDIR=$1
OUTDIR=$2
FILES=$3

if [ ! -f $FILES ]; then
    echo "$FILES doesn't exist. Exiting." 1>&2 
    exit 1
fi

if [ ! -d ${OUTDIR} ]; then
    mkdir -p ${OUTDIR}
fi

SRCDIR="${LABHOME}/roadmap/rnapred/src/rnapred"
params="--ntrees $TREES --leaf $LEAF --alphas $ALPHAS --cs $C --nproc $NPROC $NOCV"
if [[ $NOCV == "" ]]; then
    suf="cv"
    ext="npz"
else
    suf="model"
    ext="pkl"
fi

if [ $NPROC -gt 1 ]; then
    parg="-pe orte $NPROC"
else
    parg=""
fi

if [[ $MERGE -eq 1 ]] && [[ $SPLIT -eq 0 ]]; then
    # Merge experiments without excluding anything.

    script=${OUTDIR}/merged_${suf}.sh
    outfile=${OUTDIR}/merged_${suf}.${ext}
    errfile=${OUTDIR}/merged_${suf}.err
    
    command="python ${SRCDIR}/python/rnapredCV.py --method $METHOD $params --expt $FILES $INDIR $outfile"
    
    if [ $RUN -eq 1 ]; then
	echo $command
	$command
    else
	echo "#!/bin/bash" > $script    
	echo "module add python/2.7" >> $script
	echo "module add r/2.15.1" >> $script
	echo $command >> $script
	qsub -N ${sample}_${METHOD} -q standard -l h_vmem=3G -l h_rt=6:00:00 $parg -e $errfile -o /dev/null $script
    fi
else
    while read sample; do
	if [ $MERGE -eq 0 ]; then
	    pref=$sample
	    infile=${INDIR}/${pref}_feat_mat.npz
	    if [ ! -f $infile ]; then
		continue
	    fi	
	    outfile=${OUTDIR}/${pref}_${suf}.${ext}
	    command="python ${SRCDIR}/python/rnapredCV.py --method $METHOD $params $infile $outfile"  
	else
	    pref=merged_no_${sample}
	    outfile=${OUTDIR}/${pref}_${suf}.${ext}
	    command="python ${SRCDIR}/python/rnapredCV.py --method $METHOD $params"
	    command="$command --expt $FILES --exclude_expt $sample $INDIR $outfile"  
	fi
	script=${OUTDIR}/${pref}_${suf}.sh
	errfile=${OUTDIR}/${pref}_${suf}.err
	
	if [ -f $outfile ]; then
	    echo "$outfile exists. Skipping" 1>&2
	    continue
	fi

	if [ $RUN -eq 1 ]; then
	    echo $command
	    $command
	else
	    echo "#!/bin/bash" > $script    
	    echo "module add python/2.7" >> $script
	    echo "module add r/2.15.1" >> $script
	    echo $command >> $script
	    qsub -N ${pref}_${METHOD} -q standard -l h_vmem=3G -l h_rt=6:00:00 $parg -e $errfile -o /dev/null $script
	fi	
    done < $FILES
fi
