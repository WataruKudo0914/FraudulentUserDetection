DATA=$1
c1=0
while [ $c1 -le 2 ]
do
	c2=0
	while [ $c2 -le 2 ]
	do
		c3=0
		while [ $c3 -le 2 ]
		do
			c4=0
			while [ $c4 -le 2 ]
			do
				python -m src.models.rev2.run_rev2 --data-name $DATA --alpha1 $c1 --beta1 $c2\
													 --gamma1 $c3 --gamma2 $c4&
				echo $c1 $c2 $c3 $c4
				(( c4+=1 ))
                # wait;
			done
			(( c3+=1 ))
			#wait;
		done
		(( c2+=1 ))
		wait;
	done
	(( c1+=1 ))
	#wait;
done